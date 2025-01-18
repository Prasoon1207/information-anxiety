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
from transformers import BitsAndBytesConfig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] ='3' # set cuda devices

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

batch_size = 50
number_of_batches = 5
max_possible_answers = 1

import sys
model_name = sys.argv[1]
current_prop = sys.argv[2]
device = sys.argv[3]


path_to_data_directory = 'information-anxiety/data'
df = pd.read_csv(path_to_data_directory + '/' + 'data.csv').set_index('id')
df['pop'] = df['s_pop'] + df['o_pop']

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



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = quantization_config)
params = dict(model.named_parameters())    
E_proj = params['lm_head.weight']
E_bias = None

unique_prop = ['capital', 'genre', 'occupation', 'religion', 'screenwriter', 'sport']

for current_prop in unique_prop:
    
    print(current_prop)
    mlp_storer = {} 
    mlp_storer[current_prop] = {}

    subset_df = df[df['prop'] == current_prop]
    context, example_indices = create_demonstrations(subset_df, 12)
    batches = create_batches(subset_df, example_indices)

    for batch_index in tqdm(range(len(batches))):
        
        batch = batches[batch_index]
        batch_wise_average_mean = []
        batch_wise_average_std_dev = []
        
        for index, row in tqdm(batch.iterrows()):
            
            torch.cuda.empty_cache()
            
            print(index, end = " ")
            accepted_answers = ast.literal_eval(row['possible_answers'])
            accepted_answers = accepted_answers[:min(max_possible_answers, len(accepted_answers))]
            gold_tokens = []; accepted_answers_token_ids = []
            for answer in accepted_answers: accepted_answers_token_ids.append([tokenizer(answer, return_tensors="pt").input_ids[0]])
            for answer in accepted_answers_token_ids: gold_tokens.append([tokenizer.decode(answer[0][i]) for i in range(answer[0].shape[0])])
            paraphrases = list(ast.literal_eval(row['paraphrases']))
            path = 'results/' + model_name.replace('/', '-') + '/' + current_prop + '/' + str(batch_index) + '/' + str(index) + '/'
                        
            column_to_replace = accepted_answers_token_ids[0][0][1].item()
            true_distribution = np.zeros((model.config.num_hidden_layers + 1, model.config.vocab_size), dtype = 'float32')
            true_distribution[:, column_to_replace] = np.ones((model.config.num_hidden_layers + 1,))
            
            # subject_token_ids = tokenizer(row['subj'], return_tensors = "pt").input_ids.to(device)[0][1:].cpu().tolist()
            
            para_info = []
            for paraphrase_index in range(len(paraphrases)):
                paraphrase = paraphrases[paraphrase_index]
                prompt = context + 'Q:' + paraphrase + '\n'+ 'A:'
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output_hidden_states = model.generate(input_ids, num_beams = 1, num_return_sequences = 1,
                            max_new_tokens = 10, output_hidden_states = True, 
                            return_dict_in_generate = True, output_scores = True,
                            output_attentions = True, pad_token_id=tokenizer.eos_token_id).hidden_states[0]
                
                predicted_distribution = []
                projected = [torch.matmul(layer_hidden_state, E_proj.transpose(0,1)) for layer_hidden_state in output_hidden_states]
                for layer_projected in projected: predicted_distribution.append(torch.nn.functional.softmax(layer_projected, dim = -1))
                predicted_distribution = np.array([predicted_distribution[layer][0][len(input_ids[0])-1].cpu().detach().numpy() for layer in range(len(predicted_distribution))], dtype = 'float32')
                
            
                kl_divergence = get_distance(predicted_distribution, true_distribution, "kl-divergence")
                para_info.append(kl_divergence)
                
                
                
                for layer_hidden_state in output_hidden_states:
                    del layer_hidden_state
                    torch.cuda.empty_cache()
                del output_hidden_states
                model.reset_mhsa_hidden_states()
                model.reset_up_projections()
                torch.cuda.empty_cache()
                del projected; del predicted_distribution
                torch.cuda.empty_cache()

            para_info = np.array(para_info)
            mean = np.mean(para_info, axis=0)
            std_dev_values = np.std(para_info, axis = 0)
            batch_wise_average_mean.append(mean)
            batch_wise_average_std_dev.append(std_dev_values)
            
            
        batch_wise_average_mean = np.mean(np.array(batch_wise_average_mean), axis = 0)
        batch_wise_average_std_dev = np.mean(np.array(batch_wise_average_std_dev), axis = 0)
        mlp_storer[current_prop][batch_index] = (batch_wise_average_mean, batch_wise_average_std_dev)

    path_to_save = 'information-anxiety/results/distance-hidden' + '/' + model_name.replace('/','-') + '/' + current_prop + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        
    import pickle
    with open(path_to_save + 'mlp.pickle', 'wb') as handle:
        pickle.dump(mlp_storer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    mhsa_storer = {} 
    mhsa_storer[current_prop] = {}
    subset_df = df[df['prop'] == current_prop]
    context, example_indices = create_demonstrations(subset_df, 12) # creating demonstrations
    batches = create_batches(subset_df, example_indices)
    for batch_index in tqdm(range(len(batches))):

        batch = batches[batch_index]
        batch_wise_average_mean = []
        batch_wise_average_std_dev = []
        
        for index, row in batch.iterrows():
            print(index, end = " ")
            accepted_answers = ast.literal_eval(row['possible_answers'])
            accepted_answers = accepted_answers[:min(max_possible_answers, len(accepted_answers))]
            gold_tokens = []; accepted_answers_token_ids = []
            for answer in accepted_answers: accepted_answers_token_ids.append([tokenizer(answer, return_tensors="pt").input_ids[0]])
            for answer in accepted_answers_token_ids: gold_tokens.append([tokenizer.decode(answer[0][i]) for i in range(answer[0].shape[0])])
            paraphrases = list(ast.literal_eval(row['paraphrases']))
                        
            column_to_replace = accepted_answers_token_ids[0][0][1].item()
            true_distribution = np.zeros((model.config.num_hidden_layers + 1, model.config.vocab_size), dtype = 'float32')
            true_distribution[:, column_to_replace] = np.ones((model.config.num_hidden_layers + 1,))
            
            subject_token_ids = tokenizer(row['subj'], return_tensors = "pt").input_ids.to(device)[0][1:].cpu().tolist()
            para_info = []
            for paraphrase_index in range(len(paraphrases)):
                paraphrase = paraphrases[paraphrase_index]
                prompt = context + 'Q:' + paraphrase + '\n'+ 'A:'
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output = model.generate(input_ids, num_beams = 1, num_return_sequences = 1,
                            max_new_tokens = 10, output_hidden_states = True, 
                            return_dict_in_generate = True, output_scores = True,
                            output_attentions = True, pad_token_id=tokenizer.eos_token_id)
                
                mhsa_hidden_states = []
                get_mhsa_hidden_states = model.get_mhsa_hidden_states()
                model.reset_mhsa_hidden_states()
                model.reset_up_projections()
                torch.cuda.empty_cache()

                generated_token_index = 0
                while(generated_token_index < len(get_mhsa_hidden_states)):
                    mhsa_hidden_states.append(get_mhsa_hidden_states[generated_token_index: generated_token_index + model.config.num_hidden_layers])
                    generated_token_index = generated_token_index + model.config.num_hidden_layers
                
                del get_mhsa_hidden_states
                predicted_distribution = []                
                projected = [torch.matmul(layer_hidden_state, E_proj.transpose(0,1)) for layer_hidden_state in mhsa_hidden_states[0]]
                for layer_projected in projected: predicted_distribution.append(torch.nn.functional.softmax(layer_projected, dim = -1))
                predicted_distribution = np.array([predicted_distribution[layer][0][len(input_ids[0])-1].cpu().detach().numpy() for layer in range(len(predicted_distribution))], dtype = 'float32')
    
                kl_divergence = get_distance(predicted_distribution, true_distribution, "kl-divergence")
                para_info.append(kl_divergence)
                del output; del projected; del predicted_distribution
                
            para_info = np.array(para_info)
            mean = np.mean(para_info, axis=0)
            std_dev_values = np.std(para_info, axis = 0)
            batch_wise_average_mean.append(mean)
            batch_wise_average_std_dev.append(std_dev_values)
        
        batch_wise_average_mean = np.mean(np.array(batch_wise_average_mean), axis = 0)
        batch_wise_average_std_dev = np.mean(np.array(batch_wise_average_std_dev), axis = 0)
        mhsa_storer[current_prop][batch_index] = (batch_wise_average_mean, batch_wise_average_std_dev)
        
    path_to_save = 'information-anxiety/results/distance-hidden' + '/' + model_name.replace('/','-') + '/' + current_prop + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    import pickle
    with open(path_to_save + 'mhsa.pickle', 'wb') as handle:
        pickle.dump(mhsa_storer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    storer = {}
    mean_accuracy_question = {}
    storer[current_prop] = {}
    mean_accuracy_question[current_prop] = {}
    subset_df = df[df['prop'] == current_prop]
    context, example_indices = create_demonstrations(subset_df, 12) # creating demonstrations
    batches = create_batches(subset_df, example_indices)
    for batch_index in tqdm(range(len(batches))):
        
        storer[current_prop][batch_index] = {}
        mean_accuracy_question[current_prop][batch_index] = {}
        batch = batches[batch_index]
        batch_wise = [0 for index in range(model.config.num_hidden_layers)]
        
        for index, row in tqdm(batch.iterrows()):
            storer[current_prop][batch_index][index] = {}
            mean_accuracy_question[current_prop][batch_index][index] = 0
            print(index, end = " ")
            accepted_answers = ast.literal_eval(row['possible_answers'])
            accepted_answers = accepted_answers[:min(max_possible_answers, len(accepted_answers))]
            
            
            gold_tokens = []; accepted_answers_token_ids = []
            
            for answer in accepted_answers: accepted_answers_token_ids.append([tokenizer(answer, return_tensors="pt").input_ids[0]])
            for answer in accepted_answers_token_ids: gold_tokens.append([tokenizer.decode(answer[0][i]) for i in range(answer[0].shape[0])])
            del accepted_answers_token_ids
            torch.cuda.empty_cache()
            paraphrases = list(ast.literal_eval(row['paraphrases']))
            for paraphrase_index in range(len(paraphrases)):
                storer[current_prop][batch_index][index][paraphrase_index] = []
                paraphrase = paraphrases[paraphrase_index]
                prompt = context + 'Q:' + paraphrase + '\n'+ 'A:'
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                output = model.generate(input_ids, num_beams = 1, num_return_sequences = 1,
                            max_new_tokens = 10, output_hidden_states = True, 
                            return_dict_in_generate = True, output_scores = True,
                            output_attentions = True, pad_token_id=tokenizer.eos_token_id)
                
                generated_token = tokenizer.decode(output.sequences[0][len(input_ids[0])].item())
                del output
                torch.cuda.empty_cache()
                if(gold_tokens[0][1].startswith(generated_token)): mean_accuracy_question[current_prop][batch_index][index] += 1
                up_projections = []
                get_up_projections = model.get_up_projections()
                
                model.reset_up_projections()
                model.reset_mhsa_hidden_states()
                generated_token_index = 0
                
                while(generated_token_index < len(get_up_projections)):
                    up_projections.append(get_up_projections[generated_token_index: generated_token_index + model.config.num_hidden_layers])
                    generated_token_index = generated_token_index + model.config.num_hidden_layers
                for layer in range(len(up_projections[0])):
                    storer[current_prop][batch_index][index][paraphrase_index].append(up_projections[0][layer][0][len(input_ids[0]) - 1].cpu())
                
                del up_projections
                torch.cuda.empty_cache()
            mean_accuracy_question[current_prop][batch_index][index] /= len(paraphrases)

    from numpy import dot
    from numpy.linalg import norm

    def get_distance(a, b, metric):
        a = np.array(a)
        b = np.array(b)
        if(metric == 'euclidean'): return norm(a-b)
        if(metric == 'cosine'): return dot(a, b)/(norm(a) * norm(b))

    def get_similarity(a, metric):
        result = []
        for i in range(len(a)):
            for j in range(i): 
                result.append(get_distance(a[i], a[j], metric))
        return sum(result)/len(result)
        
    similarity_storer = {}
    similarity_storer[current_prop] = {}

    for batch_index in tqdm(list(storer[current_prop].keys())):
        
        similarity_storer[current_prop][batch_index] = {}
        for ques_index in list(storer[current_prop][batch_index]):
            print(ques_index, end = " ")
            layer_wise_similarity = [0 for _ in range(model.config.num_hidden_layers)]
            
            for layer_index in range(model.config.num_hidden_layers):
                layer_wise_vectors = []
                for paraphrase_index in range(len(storer[current_prop][batch_index][ques_index])):
                    layer_wise_vectors.append(storer[current_prop][batch_index][ques_index][paraphrase_index][layer_index])
                layer_wise_similarity[layer_index] = get_similarity(layer_wise_vectors, "cosine")
            similarity_storer[current_prop][batch_index][ques_index] = layer_wise_similarity
        
    path_to_save = 'information-anxiety/results/neural' + '/' + model_name.replace('/','-') + '/' + current_prop + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    import pickle
    with open(path_to_save + 'sim.pickle', 'wb') as handle:
        pickle.dump(similarity_storer, handle, protocol=pickle.HIGHEST_PROTOCOL)