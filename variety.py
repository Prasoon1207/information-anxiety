import json
import math
import os
import random
import re
import time
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM,StoppingCriteria, StoppingCriteriaList)

import sys

model_name = sys.argv[1]
device = sys.argv[2]

batch_size = 20
number_of_batches = 25
n_outputs = 1


def create_demonstrations(subset_df, n_shots):
    context = "Answer following questions in one word or phrase.\n"
    example_indices = subset_df.sample(n = n_shots).index
    for index in example_indices:
        row = subset_df.loc[index]
        context += 'Q:' + row['question'] + '\n' + 'A:' + row['possible_answers'][1:-1].split(', ')[0][1: -1] + '\n'
    return (context, example_indices)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  
def normalize_text(s):
    import re
    import string
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_start_score(prediction_start_token, truth):
    return int(normalize_text(truth).startswith(normalize_text(prediction_start_token)))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log10(p)
    return entropy

def create_batches(subset_df, example_indices, batch_size = batch_size, number_of_batches = number_of_batches):
    remaining_indices = subset_df.index.difference(example_indices) # removing questions from the experimental setup that were used for demonstrations
    left_df = subset_df.loc[remaining_indices].sort_values(by = 'pop', ascending = False)
    print(left_df.shape)
    number_of_batches_init = left_df.shape[0] // batch_size
    if(number_of_batches_init < number_of_batches):
        chosen_batches = [i for i in range(0, number_of_batches_init, 1)]
    else:
        chosen_batches = [i for i in range(0, (number_of_batches_init//number_of_batches)*number_of_batches, number_of_batches_init//number_of_batches)]
    batches = []
    for index in chosen_batches: # left_df is sorted in decreasing levels of popularity
        start_idx = index * batch_size
        end_idx = start_idx + batch_size
        batch = left_df.iloc[start_idx:end_idx] # each batch is a pandas DataFrame
        batches.append(batch)
    return batches

set_seed(42)

prev = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit = True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


newline_id = tokenizer.convert_tokens_to_ids('\n')
print("...Model has been retrieved from the library")
print("Time elapsed from previous checkpoint: ", time.perf_counter() - prev)


path_to_data_directory = 'information-anxiety/data'
df = pd.read_csv(path_to_data_directory + '/' + 'data.csv').set_index('id')
paraphraser = pd.read_csv(path_to_data_directory + '/' + 'paraphraser.csv').set_index('prop')
df['pop'] = df['s_pop'] + df['o_pop']



unique_prop = ['capital', 'genre', 'occupation', 'religion', 'screenwriter', 'sport']
for current_prop in unique_prop:
    print(current_prop)
    subset_df = df[df['prop'] == current_prop]
    
    context, example_indices = create_demonstrations(subset_df, 12)
    batches = create_batches(subset_df, example_indices)
    columns = list(subset_df.columns)
    columns.extend(['em_scores', 'start_scores', 'f1_scores', 'uncertainty_metric', 'prob'])
    
    for batch_index in range(len(batches)):
        result = pd.DataFrame(columns = columns)
        batch = batches[batch_index]
        for index, row in tqdm(batch.iterrows()):
            
            acc_f1 = []
            acc_em = []
            acc_st = []
            unc = []
            prob = []
            
            for current_question in ast.literal_eval(row['paraphrases']):
                
                encoder_input_str = context + 'Q:' + current_question + '\n'+ 'A:'
                expected_answers = row['possible_answers'][1:-1]
                expected_answers = re.findall(r'"(.*?)"', expected_answers)
                inputs = tokenizer(encoder_input_str, return_tensors="pt").input_ids
                inputs = inputs.to(device)
                outputs = model.generate(
                    inputs,
                    num_beams=1,
                    num_return_sequences = n_outputs,
                    no_repeat_ngram_size = 2,
                    max_new_tokens = 10,
                    return_dict_in_generate = True, 
                    output_scores=True,
                )
                
                model.reset_mhsa_hidden_states()
                model.reset_up_projections()
                    
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores, 
                    normalize_logits = True
                )
                
                predicted_answers = []
                
                for index in range(n_outputs):
                    generated_token_ids = outputs.sequences[index].tolist()
                    input_length = len(inputs.tolist()[0])
                    context_tokens = inputs[0][:input_length].tolist()
                    generated_tokens_ids = generated_token_ids[input_length:]
                    context_output = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    generated_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
                    predicted_answers.append(generated_text)
                    
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_tokens_ids)
                
                input_length = 1 if model.config.is_encoder_decoder else inputs.shape[1]
                output_length = input_length + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
                length_penalty = model.generation_config.length_penalty
                
                reconstructed_scores = transition_scores.cpu().sum(axis=1) / (output_length**length_penalty)
                
                f1_score = max((compute_f1(predicted_answers[0], answer)) for answer in expected_answers)
                em_score = max((compute_exact_match(predicted_answers[0], answer)) for answer in expected_answers)
                st_score = max((compute_start_score(generated_tokens[0], answer)) for answer in expected_answers) 
                
                f1_score = max((compute_f1(predicted_answers[0], answer)) for answer in expected_answers)
                em_score = max((compute_exact_match(predicted_answers[0], answer)) for answer in expected_answers)
                acc_f1.append(f1_score)
                acc_em.append(em_score)
                acc_st.append(st_score)
                generated_logits = (transition_scores.cpu().numpy())[0][1:]
                uncertainty = entropy(list(np.exp(generated_logits)))
                probability = np.exp(np.sum(generated_logits))
                
                
                assert(probability <= 1)
                unc.append(uncertainty)
                prob.append(probability)
            
            current_row = row.to_dict()
            current_row['em_scores'] = acc_em
            current_row['f1_scores'] = acc_f1
            current_row['st_scores'] = acc_st
            current_row['uncertainty_metric'] = unc
            current_row['prob'] = prob
            result.loc[len(result.index)] = current_row
            
        
        print(batch_index)   
        saving_path = 'information-anxiety/results/variety' + '/' + model_name.replace('/', '-') + '/' + current_prop + '/'
        if(os.path.exists(saving_path) == False):
            os.makedirs(saving_path)
        result.to_csv(saving_path + str(batch_index) + '.csv', index = False)

