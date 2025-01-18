import os
import sys
import ast
import math
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import os
from transformers import BitsAndBytesConfig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] ='1,3'

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

batch_size = 50
number_of_batches = 5
max_possible_answers = 3


model_name = sys.argv[1]
device = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
params = dict(model.named_parameters())
    
    
df = pd.read_csv('/home/prasoon/snap/adaptive-retrieval-main/data/popQA.csv')
df['pop'] = df['s_pop'] + df['o_pop']

all_prop_unchanged = list(df['prop'].unique())

def create_demonstrations(subset_df, n_shots):
    context = "Answer following questions in one word or phrase."
    example_indices = subset_df.sample(n = n_shots).index
    for index in example_indices:
        row = subset_df.loc[index]
        context += 'Q:' + row['question'] + '\n' + 'A:' + row['possible_answers'][1:-1].split(', ')[0][1: -1] + '\n'
    return (context, example_indices)

def get_distance(predicted_distribution, true_distribution, mode, eps = 1e-16):
    distance = []
    for index in range(predicted_distribution.shape[0]):
        p = true_distribution[index] + eps
        q = predicted_distribution[index] + eps
        distance.append(np.sum(p*(np.log(p) - np.log(q))))
    return distance

def create_batches(current_prop, subset_df, example_indices, batch_size = batch_size, number_of_batches = number_of_batches):
    remaining_indices = subset_df.index.difference(example_indices) # removing questions from the experimental setup that were used for demonstrations
    left_df = subset_df.loc[remaining_indices].sort_values(by = 'pop', ascending = False)
    print(left_df.shape)
    number_of_batches_init = left_df.shape[0] // batch_size
    # assert(number_of_batches_init >= number_of_batches)
    if(number_of_batches_init < number_of_batches):
        return -1
    chosen_batches = [i for i in range(0, (number_of_batches_init//number_of_batches)*number_of_batches, number_of_batches_init//number_of_batches)]
    batches = []
    for index in chosen_batches: # left_df is sorted in decreasing levels of popularity
        start_idx = index * batch_size
        end_idx = start_idx + batch_size
        batch = left_df.iloc[start_idx:end_idx] # each batch is a pandas DataFrame
        batches.append(batch)
    return batches

        

E_proj = params['lm_head.weight']
E_bias = None

storer = {}
all_prop = []
for current_prop in all_prop_unchanged:

    print(current_prop)
    storer[current_prop] = {}
    subset_df = df[df['prop'] == current_prop]
    context, example_indices = create_demonstrations(subset_df, 12)
    batches = create_batches(current_prop, subset_df, example_indices)
    if(batches == -1):
        continue
    all_prop.append(current_prop)
    for batch_index in tqdm(range(len(batches))):
        print(batch_index)
        storer[current_prop][batch_index] = {}
        batch = batches[batch_index]
        batch_wise = [0 for index in range(model.config.num_hidden_layers)]
        
        for index, row in batch.iterrows():
            
            storer[current_prop][batch_index][index] = []
            print(index, end = " ")
            prompt = context + 'Q:' + row['question'] + '\n' + 'A:'
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            output = model.generate(input_ids, num_beams = 1, num_return_sequences = 1,
                            max_new_tokens = 10, output_hidden_states = True, 
                            return_dict_in_generate = True, output_scores = True,
                            output_attentions = True, pad_token_id=tokenizer.eos_token_id)
            
            up_projections = []
            get_up_projections = model.get_up_projections()
            model.reset_up_projections()
            model.reset_mhsa_hidden_states()
            torch.cuda.empty_cache()

            generated_token_index = 0
            while(generated_token_index < len(get_up_projections)):
                up_projections.append(get_up_projections[generated_token_index: generated_token_index + model.config.num_hidden_layers])
                generated_token_index = generated_token_index + model.config.num_hidden_layers
            for layer in range(len(up_projections[0])):
                storer[current_prop][batch_index][index].append(up_projections[0][layer][0][len(input_ids[0]) - 1].cpu())
                
                
import pickle

path_to_save = '/home/prasoon/snap/main/results/matrix/' + model_name.replace('/', '-') + '/'
if(os.path.exists(path_to_save) == False):
        os.makedirs(path_to_save)
with open(path_to_save + 'storer.pkl', 'wb') as handle:
    pickle.dump(storer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_to_save + 'storer.pkl', 'rb') as handle:
    storer = pickle.load(handle)
print("pickle retrieved.....")
    
all_prop = []
for prop in list(storer.keys()):
    if(len(list(storer[prop].keys())) != 0):
        all_prop.append(prop)
        
    
from numpy import dot
from numpy.linalg import norm


# num_hidden_layers = model.config.num_hidden_layers
num_hidden_layers = 80
def get_distance(a, b, metric):
    a = np.array(a)
    b = np.array(b)
    if(metric == 'euclidean'): return norm(a-b)
    if(metric == 'cosine'): return dot(a, b)/(norm(a) * norm(b))

matrix = {}
for batch_index in range(number_of_batches):
    matrix[batch_index] = {}
    for prop_1 in all_prop:
        matrix[batch_index][prop_1] = {}
    
    for prop_1 in all_prop:  
        for prop_2 in all_prop:
            matrix[batch_index][prop_1][prop_2] = {}
            
for batch_index in tqdm([0, number_of_batches-1]):
    for prop_1 in all_prop:
        for prop_2 in all_prop:
            print(prop_1, prop_2)
            
            layer_wise_util = {}
            for layer_index in range(num_hidden_layers):
                layer_wise_util[layer_index] = []
            
            for ques_a in tqdm(list(storer[prop_1][batch_index])):
                for ques_b in tqdm(list(storer[prop_2][batch_index])):
                    for layer_index in range(num_hidden_layers):
        
                        layer_wise_util[layer_index].append(get_distance(storer[prop_1][batch_index][ques_a][layer_index],
                                storer[prop_2][batch_index][ques_b][layer_index], 'cosine'))
            
                       
            for layer_index in range(num_hidden_layers):
                matrix[batch_index][prop_1][prop_2][layer_index] = (np.mean(layer_wise_util[layer_index]), 
                                                                    np.std(layer_wise_util[layer_index]))
                
                    

import pandas as pd

def ratio_diag(arr):

    arr = arr.to_numpy()
    
    non_diag_mask = ~np.eye(arr.shape[0], dtype=bool)
    diag_mask = np.eye(arr.shape[0], dtype=bool)
    
    non_diagonal_elements = arr[non_diag_mask]
    diagonal_elements = arr[diag_mask]
    
    avg_non_diagonal = np.mean(non_diagonal_elements)
    avg_diagonal = np.mean(diagonal_elements)
    
    return avg_non_diagonal/avg_diagonal

def average_non_diagonal_elements(arr):

    diagonal_elements = np.diag(arr)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    non_diagonal_elements = arr[mask]
    avg_non_diagonal = np.mean(non_diagonal_elements)

    return avg_non_diagonal

def diagonality(arr):
    d = arr.shape[0]
    j = np.array([1 for index in range(d)])
    r = np.array([(index+1) for index in range(d)])
    r2 = np.array([(index+1)**2 for index in range(d)])
    
    n = j @ arr @ j.T
    sum_x = r @ arr @ j.T
    sum_y = j @ arr @ r.T
    sum_x2 = r2 @ arr @ j.T
    sum_y2 = j @ arr @ r2.T
    sum_xy = r @ arr @ r.T
    
    return (n*sum_xy - (sum_x)*(sum_y))/(math.sqrt(n*sum_x2 - (sum_x)**2)*math.sqrt(n*sum_y2 - (sum_y)**2))

batch_wise_similarity = {}
for batch_index in [0, number_of_batches-1]:
    batch_wise_similarity[batch_index] = []
    layer_wise_image_mean = {}
    layer_wise_image_std = {}
    for layer_index in range(num_hidden_layers):
        
        layer_wise_image_mean[layer_index] = pd.DataFrame(index=all_prop, columns=all_prop)
        layer_wise_image_std[layer_index] = pd.DataFrame(index=all_prop, columns=all_prop)
        
        for prop_a in all_prop:
            for prop_b in all_prop:
                
                layer_wise_image_mean[layer_index].loc[prop_a, prop_b] = matrix[batch_index][prop_a][prop_b][layer_index][0]
                layer_wise_image_mean[layer_index].loc[prop_a, prop_b] = matrix[batch_index][prop_a][prop_b][layer_index][0]

    for layer_index in range(num_hidden_layers):
        sns_plot = sns.heatmap(layer_wise_image_mean[layer_index].astype(float), annot=True, cmap =sns.cm.rocket_r,
                            linecolor='white', linewidths=1)
        batch_wise_similarity[batch_index].append(ratio_diag(layer_wise_image_mean[layer_index].astype(float)))
        sns_plot.set_title(str(layer_index) + " " + str(ratio_diag(layer_wise_image_mean[layer_index].astype(float))))
        path_to_save = '/home/prasoon/snap/main/results/matrix/'+model_name.replace('/', '-')+'/'+str(batch_index)+'/'
        if(os.path.exists(path_to_save) == False):
            os.makedirs(path_to_save)
        plt.savefig(path_to_save + str(layer_index), dpi=800)
        plt.clf()

import imageio
ims = []
for batch_index in [0, number_of_batches-1]:
    path_to_directory = '/home/prasoon/snap/main/results/matrix/'+model_name.replace('/', '-') +'/'+str(batch_index)+'/'
    im_paths = [path_to_directory+str(layer_index)+'.png' for layer_index in range(num_hidden_layers)]
    for im_path in im_paths:
        ims.append(imageio.imread(im_path))
    path_to_save_gifs = '/home/prasoon/snap/main/results/matrix/' + model_name.replace('/', '-') + '/' + str(batch_index) + '.gif'
    imageio.mimwrite(path_to_save_gifs, ims, format='GIF', duration = 5)
    
cmap = 'plasma'
clrs = sns.color_palette(cmap, number_of_batches)
labels = {0: 'High Popularity', number_of_batches-1: 'Low Popularity'}

for batch_index in [0, number_of_batches-1]:
    plt.plot([layer_index for layer_index in range(num_hidden_layers)], batch_wise_similarity[batch_index], marker='o', 
             linestyle='-', c = clrs[batch_index], label = labels[batch_index])

    plt.xlabel('layers', fontsize=12, color='green')
    plt.ylabel('ratio-diag', fontsize=12, color='blue')
    plt.title(model_name, fontsize=12)
    plt.legend()
    
plt.savefig('/home/prasoon/snap/main/results/matrix/' + model_name.replace('/', '-') + '/' + 'diagonal.png', dpi = 1200)

