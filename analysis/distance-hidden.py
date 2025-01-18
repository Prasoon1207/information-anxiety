import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import pickle


number_of_batches = 5

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
    parser.add_argument("-c", "--current_prop", help="name of the property", required=True)
    parser.add_argument("-i", "--path_to_results", help="path of the directory containing input data files", required=True)
    parser.add_argument("-m", "--model_name", help="name of the model", required=True)
    args = parser.parse_args()
    
    model = args.model_name
    path_to_save = args.path_to_save
    path_to_results = args.path_to_results
    current_prop = args.current_prop
    
    for mode in ['mlp', 'mhsa']:

        with open(path_to_results + '/' + model + '/' + current_prop + '/' + mode + '.pickle', 'rb') as handle:
            storer = pickle.load(handle)

        fig, ax = plt.subplots()
        fontsize = 14
        clrs = sns.color_palette("viridis", 5)
        labels = {0: 'Head', number_of_batches//2: 'Torso', number_of_batches-1: "Tail"}
        for batch_index in [0, number_of_batches//2 , number_of_batches-1]:
            
            with sns.axes_style("darkgrid"):
                epochs = list(range(number_of_layers[model]))
                if(mode == 'mlp'):
                    epochs = list(range(number_of_layers[model]+1))
                
                mean = np.array(storer[current_prop][batch_index][0], dtype=np.float64)
                std = np.array(storer[current_prop][batch_index][1], dtype=np.float64)
                ax.plot(epochs, mean, label = labels[batch_index], c=clrs[batch_index], linewidth = 1.5)
                ax.fill_between(epochs, mean-std, mean+std ,alpha=0.2, facecolor=clrs[batch_index])
                ax.legend(fontsize = fontsize, loc = 'upper right')
                # ax.set_ylabel('Distance between true distribution and \ninverted predicted distribution', fontsize = fontsize)
                # ax.set_xlabel('Layer', fontsize = fontsize)
                # ax.grid(which='major', axis='x', linestyle='--')
                # ax.xaxis.grid(True, which='major', linestyle=(0, (10, 15)))
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                ax.xaxis.grid(True, color = 'white')
                ax.set_facecolor((0.95, 0.95, 0.95))
        
        if(os.path.exists(path_to_save + '/' + model + '/' + current_prop + '/') == False):
            os.makedirs(path_to_save + '/' + model + '/' + current_prop + '/')

        fig.savefig(path_to_save + '/' + model + '/' + current_prop + '/' + mode + '.svg', format = 'svg', bbox_inches = 'tight')
        # fig.savefig(path_to_save + '/' + model + '/' + current_prop + '/' + mode + '.pdf', format = 'pdf', bbox_inches = 'tight')