import os
import argparse
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    number_of_batches = 3
    number_of_paraphrases = 10
    result_dict = {}
    
    with open(path_to_results + '/' + model + '/' + current_prop + '/' + 'heat.pickle', 'rb') as handle:
        heat_storer = pickle.load(handle)

    for batch_index in [0, int(number_of_batches/2), number_of_batches-1]:
        result = pd.DataFrame(columns = ['layer_index', 'mean', 'std'])
        
        result_std  = []
        for layer_index in range(number_of_layers[model]):
            
            for ques_index in list(heat_storer[current_prop][batch_index].keys()):
                store = []
                for paraphrase_index in range(number_of_paraphrases):
                    store.append(heat_storer[current_prop][batch_index][ques_index][paraphrase_index][layer_index])
            store = np.array(store)
            result.loc[len(result)] = [layer_index, np.mean(store), np.std(store)]
            
        result_dict[batch_index] = result
        
    
    xval = np.arange(0.1, 4, 0.5)
    yval = np.exp(-xval)

    # Customize error bar styles
    errorbar_kwargs = dict(lw=1, capsize=3, capthick=2, elinewidth=2, markeredgewidth=2)
    epochs = [index + 1 for index in range(number_of_layers[model])]
    clrs = sns.color_palette("viridis", number_of_batches)
    labels = {0: "Head", int(number_of_batches/2): "Torso", number_of_batches-1: "Tail"}
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.xaxis.grid(True, color='white', linewidth=1)
    ax.set_facecolor((0.95, 0.95, 0.95))
    
    fontsize = 14
    
    for batch_index in [0, number_of_batches-1]:
        plt.errorbar(epochs, result_dict[batch_index]['mean'], yerr=result_dict[batch_index]['std'], color=clrs[batch_index], fmt='o', **errorbar_kwargs, ms = 1)
        plt.plot(epochs, result_dict[batch_index]['mean'], color=clrs[batch_index], linewidth = 1.5, label = labels[batch_index])

    yticks = [i for i in range(-8, 24, 3)]
    plt.yticks(yticks)

    
    plt.legend(fontsize = fontsize + 7, loc = 'upper right')
    plt.xticks(fontsize = fontsize + 5)
    plt.yticks(fontsize = fontsize + 5)
    
    if(os.path.exists(path_to_save + '/' + model + '/' + current_prop + '/') == False):
        os.makedirs(path_to_save + '/' + model + '/' + current_prop + '/')
    plt.savefig(path_to_save + '/' + model + '/' + current_prop + '/' +'plot.svg', format = 'svg', bbox_inches='tight')