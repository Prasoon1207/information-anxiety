import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import util_plotter
import numpy as np
import argparse
import pickle
import os

number_of_batches = 5
props = ['occupation', 'place of birth', 'genre', 'father', 'screenwriter', 'composer', 'color', 'religion', 'sport', 'author', 'mother', 'capital']
number_of_layers = {
    "meta-llama-Llama-2-7b-chat-hf": 32,
    "meta-llama-Llama-2-7b-hf": 32,
    "meta-llama-Llama-2-13b-chat-hf": 40,
    "meta-llama-Llama-2-13b-hf": 40,
    "meta-llama-Llama-2-70b-hf": 80
}

if __name__ == '__main__':
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--path_to_save", help="path of the directory to save output plots", required=True)
    parser.add_argument("-i", "--path_to_results", help="path of the directory containing input data files", required=True)
    parser.add_argument("-m", "--model_name", help="name of the model", required=True)
    args = parser.parse_args()
    
    model = args.model_name
    path_to_save = args.path_to_save
    path_to_results = args.path_to_results
    
    num_hidden_layers = number_of_layers[model]
    
    with open(path_to_results + '/' + model + '/storer.pkl', 'rb') as handle:
        storer = pickle.load(handle)
    all_prop = []
    for prop in props:
        if(len(storer[prop].keys()) > 0):
            all_prop.append(prop)
            
        
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
                
                layer_wise_util = {}
                for layer_index in range(num_hidden_layers):
                    layer_wise_util[layer_index] = []
                
                for ques_a in list(storer[prop_1][batch_index]):
                    for ques_b in list(storer[prop_2][batch_index]):
                        for layer_index in range(num_hidden_layers):
            
                            layer_wise_util[layer_index].append(util_plotter.get_distance(storer[prop_1][batch_index][ques_a][layer_index],
                                    storer[prop_2][batch_index][ques_b][layer_index], 'cosine'))
                
                for layer_index in range(num_hidden_layers):
                    matrix[batch_index][prop_1][prop_2][layer_index] = (np.mean(layer_wise_util[layer_index]), 
                                                                        np.std(layer_wise_util[layer_index]))
                    


        

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
            # sns_plot = sns.heatmap(layer_wise_image_mean[layer_index].astype(float), annot=True, cmap =sns.cm.rocket_r,
            #                     linecolor='white', linewidths=1)
            
            sns_plot = sns.heatmap(layer_wise_image_mean[layer_index].astype(float), annot=True,
                                linecolor='white', linewidths=0.5)
            batch_wise_similarity[batch_index].append(util_plotter.ratio_diag(layer_wise_image_mean[layer_index].astype(float)))
            sns_plot.set_title(str(layer_index) + " " + str(util_plotter.ratio_diag(layer_wise_image_mean[layer_index].astype(float))))
            
            if(os.path.exists(path_to_save + '/' + model + '/' + str(batch_index) + '/') == False):
                os.makedirs(path_to_save + '/' + model + '/' + str(batch_index) + '/')
                
            plt.savefig(path_to_save + '/' + model + '/' + str(batch_index) + '/' + str(layer_index) + '.svg', format = 'svg', bbox_inches = 'tight')
            plt.clf()
        
    cmap = 'plasma'
    clrs = sns.color_palette(cmap, number_of_batches)
    labels = {0: 'High Popularity', number_of_batches-1: 'Low Popularity'}
    fontsize = 14
    for batch_index in [0, number_of_batches-1]:
        
        plt.plot([layer_index for layer_index in range(num_hidden_layers)], batch_wise_similarity[batch_index], marker='o', 
                linestyle='-', c = clrs[batch_index], label = labels[batch_index])

        # plt.xlabel('layers', fontsize=fontsize, color='green')
        # plt.ylabel('ratio-diag', fontsize=fontsize, color='blue')
        plt.title(model, fontsize=fontsize)
        plt.legend()

    plt.savefig(path_to_save + '/' + model + '/' + 'diagonal.svg', format = 'svg', bbox_inches = 'tight')
