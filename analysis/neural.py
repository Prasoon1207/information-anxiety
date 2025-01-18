import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import os
import argparse
import pickle
import time



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
    current_prop = args.current_prop
    path_to_results = args.path_to_results
    path_to_save = args.path_to_save
    
    number_of_batches = 5
    
    start = time.time()
    with open(path_to_results + '/' + model + '/' + current_prop + '/' + 'sim.pickle', 'rb') as handle:
        similarity_storer  = pickle.load(handle)
    # print("Pickle file extracted in ->", time.time() - start)

    # diagram specification
    fig, ax = plt.subplots()
    ax.xaxis.grid(True, color='white', linewidth=1)
    ax.set_facecolor((0.95, 0.95, 0.95))
    fontsize = 14
    cmap = 'viridis'
    cmap_correctness = 'viridis'
    clrs = sns.color_palette(cmap, number_of_batches)
    clrs_correctness = sns.color_palette(cmap_correctness, 11)



    for batch_index in range(number_of_batches):
        
        batch_wise = [[] for _ in range(number_of_layers[model])]
        for ques_index in list(similarity_storer[current_prop][batch_index].keys()):
            
            for layer_index in range(number_of_layers[model]):
                batch_wise[layer_index].append(similarity_storer[current_prop][batch_index][ques_index][layer_index])
        
        batch_wise_mean = [None for _ in range(number_of_layers[model])]
        batch_wise_std = [None for _ in range(number_of_layers[model])]
        
        for layer_index in range(number_of_layers[model]):
            batch_wise_mean[layer_index] = np.mean(batch_wise[layer_index])
            batch_wise_std[layer_index] = np.std(batch_wise[layer_index])
            
        batch_wise_mean = np.array(batch_wise_mean)
        batch_wise_std = np.array(batch_wise_std)
        # with sns.axes_style("darkgrid"):
        #     epochs = list(range(number_of_layers[model]))
        #     ax.plot(epochs, batch_wise_mean, c=clrs[batch_index])
        epochs = list(range(number_of_layers[model]))
        plt.plot(epochs, batch_wise_mean, c=clrs[batch_index])
        yticks = plt.yticks()[0]
        plt.yticks(yticks, [f'{tick:.2f}' for tick in yticks])
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        
    norm = plt.Normalize(0, number_of_batches - 1)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[0, number_of_batches -1])
    cbar.ax.set_yticklabels(['High \nPopularity', 'Low \nPopularity'])

    # ax.set(xlabel='Layers', ylabel='Similarity across paraphrases \nof neurons using cosine similarity')
    
    if(os.path.exists(path_to_save + '/' + model + '/') == False):
        os.makedirs(path_to_save + '/' + model + '/')
    fig.savefig(path_to_save + '/' + model + '/' + current_prop + '.svg', format = 'svg', bbox_inches = 'tight')
    plt.close()

