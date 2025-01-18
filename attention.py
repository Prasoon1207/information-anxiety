import os
import sys
import ast
import math
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] ='1' # select device

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


batch_size = 50
number_of_batches = 3
max_possible_answers = 1
number_of_paraphrases = 10


import sys
model_name = sys.argv[1]
current_prop = sys.argv[2]
device = sys.argv[3] 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = quantization_config)
params = dict(model.named_parameters())        

E_proj = params['lm_head.weight']
E_bias = None

def create_batches(subset_df, example_indices, batch_size = batch_size, number_of_batches = number_of_batches):
    remaining_indices = subset_df.index.difference(example_indices) # removing questions from the experimental setup that were used for demonstrations
    left_df = subset_df.loc[remaining_indices].sort_values(by = 'pop', ascending = False)
    print(left_df.shape)
    number_of_batches_init = left_df.shape[0] // batch_size
    assert(number_of_batches_init >= number_of_batches)
    chosen_batches = [i for i in range(0, (number_of_batches_init//number_of_batches)*number_of_batches, number_of_batches_init//number_of_batches)]
    batches = []
    for index in chosen_batches: # left_df is sorted in decreasing levels of popularity
        start_idx = index * batch_size
        end_idx = start_idx + batch_size
        batch = left_df.iloc[start_idx:end_idx] # each batch is a pandas DataFrame
        batches.append(batch)
    return batches

def create_demonstrations(subset_df, n_shots):
    context = "Answer following questions in one word or phrase.\n"
    example_indices = subset_df.sample(n = n_shots).index
    for index in example_indices:
        row = subset_df.loc[index]
        context += 'Q:' + row['question'] + '\n' + 'A:' + row['possible_answers'][1:-1].split(', ')[0][1: -1] + '\n'
    return (context, example_indices)

def find_word_start_index(tokens, target_word, start_search_index):
    
    constraint_token_indices = []
    sentence_tokens = []
    for token in tokens:
        sentence_tokens.append(token.replace('▁', ''))
    
    current_index = start_search_index
    
    while(target_word.startswith(sentence_tokens[current_index]) == False):
        current_index += 1
    
    # target_word is starting with sentence word current_index
    current_string = ''
    for index in range(current_index, len(sentence_tokens)):
        current_string += sentence_tokens[index]
        if(target_word.startswith(current_string) == False):
            break
        constraint_token_indices.append(index)
    return (constraint_token_indices, index)

path_to_data_directory = 'information-anxiety/data'
df = pd.read_csv(path_to_data_directory + '/' + 'data.csv').set_index('id')
paraphraser = pd.read_csv(path_to_data_directory + '/' + 'paraphraser.csv').set_index('prop')
df['pop'] = df['s_pop'] + df['o_pop']

unique_prop = ['capital', 'genre', 'occupation', 'religion', 'screenwriter', 'sport']

for current_prop in unique_prop:
    heat_storer = {}
    heat_storer[current_prop] = {}
    subset_df = df[df['prop'] == current_prop]
    context, example_indices = create_demonstrations(subset_df, 0)
    print(context)
    batches = create_batches(subset_df, example_indices)

    for batch_index in tqdm([0, int(number_of_batches/2), number_of_batches-1]):
        heat_storer[current_prop][batch_index] = {}
        batch = batches[batch_index]
        print(batch_index, batch.shape)
        for index, row in batch.iterrows():
            print(index, end = " ")
            heat_storer[current_prop][batch_index][index] = {}
            paraphrases = list(ast.literal_eval(row['paraphrases']))
            constr_words = list(ast.literal_eval(paraphraser.loc[current_prop]['constraint_tokens']))
            print(constr_words)
            
            for paraphrase_index in range(len(paraphrases)):
                
                paraphrase = paraphrases[paraphrase_index].replace('<MASK>', row['subj'])
                prompt = context + 'Q:' + paraphrase + '\n'+ 'A:'
                
                constraint_token_indices = []

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                next_start_index = 0

                for word in constr_words[paraphrase_index]:
                    
                    (current_word_ids, next_start_index) = find_word_start_index(tokens, word, next_start_index)
                    constraint_token_indices += current_word_ids
                
                next_start_index = 0
                
                for word in row['subj'].split(' '):
                    (current_word_ids, next_start_index) = find_word_start_index(tokens, word, next_start_index)
                    constraint_token_indices += current_word_ids

                output = model.generate(input_ids, num_beams = 1, num_return_sequences = 1,
                            max_new_tokens = 10, output_hidden_states = True, 
                            return_dict_in_generate = True, output_scores = True,
                            output_attentions = True)
                
                model.reset_mhsa_hidden_states()
                model.reset_up_projections()

                    
                heat_layers = [-1 for layer_index in range(model.config.num_hidden_layers)]
                for layer_index in range(model.config.num_hidden_layers):
                    max_across_constraints = -1
                    for constraint_token_index in constraint_token_indices:
                        heat_cins = 0
                        for head_index in range(model.config.num_attention_heads):
                            heat_cins += output.attentions[0][layer_index][0][head_index][-1][constraint_token_index].item()
                        max_across_constraints = max(max_across_constraints, heat_cins)
                    heat_layers[layer_index] = max(heat_layers[layer_index], max_across_constraints)      
                heat_storer[current_prop][batch_index][index][paraphrase_index] = heat_layers
        
    path_to_save = '/home/prasoon/snap/main/results/'+model_name+'/'+current_prop+'/CSP/'
    if(os.path.exists(path_to_save) == False):
        os.makedirs(path_to_save)    
    import pickle
    with open(path_to_save + 'heat.pickle', 'wb') as handle:
        pickle.dump(heat_storer, handle, protocol=pickle.HIGHEST_PROTOCOL)