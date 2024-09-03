
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import statistics
import ast
import math
import pickle



def variety_plot(prop_wise_result, path_to_save):
    
    popularity = []
    variety = []
    mean_max_uncertainty = []
    mean_max_accuracy = []
    accuracy_error = []
    uncertainty_error = []
    
    for index in range(len(prop_wise_result)):
        
        popularity.append(prop_wise_result[index][0])
        variety.append(prop_wise_result[index][1])
        mean_max_uncertainty.append(prop_wise_result[index][2])
        mean_max_accuracy.append(prop_wise_result[index][3])
        accuracy_error.append(prop_wise_result[index][4])
        uncertainty_error.append(prop_wise_result[index][5])
        

    x_pos = np.arange(len(popularity))
    
    fig, ax = plt.subplots(2, 2, figsize = (20,10))
    ax[0, 0].scatter(popularity, mean_max_accuracy, color='blue')
    ax[0, 0].set_ylabel('Mean Max Acc. (F1-Score) per question batch-wise')
    ax[0, 0].set_xlabel('log popularity')
    ax[0, 0].xaxis.grid(True)
    
    ax[0, 1].scatter(popularity, accuracy_error, color = 'blue')
    ax[0, 1].set_ylabel('Stdev Max Acc. (F1-Score) per question batch-wise')
    ax[0, 1].set_xlabel('log popularity')
    ax[0, 1].xaxis.grid(True)




    ax[1, 0].scatter(popularity, mean_max_uncertainty, color='blue')
    ax[1, 0].set_ylabel('Mean Max Uncertainty per question batch-wise')
    ax[1, 0].set_xlabel('log popularity')
    ax[1, 0].xaxis.grid(True)

    ax[1, 1].scatter(popularity, uncertainty_error, color = 'blue')
    ax[1, 1].set_ylabel('Stdev Max Uncertainty per question batch-wise')
    ax[1, 1].set_xlabel('log popularity')
    ax[1, 1].xaxis.grid(True)

    plt.tight_layout()
    plt.savefig(path_to_save)


def process(li):
    # function to convert a string encoded list to a list data-type
    return list(map(ast.literal_eval, li))



def extract_results(df):    
    
    f1_score = process(list(df['f1_scores'])) 
    exact_match_accuracy = process(list(df['em_scores']))
    uncertainty = process(list(df['uncertainty_metric']))
    
    
    variation_f1 = list(map(statistics.stdev, f1_score))
    
    max_accuracy_per_question = [max(f1_score[index]) for index in range(len(f1_score))]
    mean_accuracy_per_question = [statistics.mean(f1_score[index]) for index in range(len(f1_score))]
    std_accuracy_per_question = [statistics.stdev(f1_score[index]) for index in range(len(f1_score))]
    
    max_uncertainty_per_question = [max(uncertainty[index]) for index in range(len(uncertainty))]
    mean_uncertainty_per_question = [statistics.mean(uncertainty[index]) for index in range(len(uncertainty))]
    std_uncertainty_per_question = [statistics.stdev(uncertainty[index]) for index in range(len(uncertainty))]
    
    variety = statistics.mean(variation_f1)
    mean_max_accuracy = statistics.mean(max_accuracy_per_question)
    mean_max_uncertainty = statistics.mean(max_uncertainty_per_question)
    
    accuracy_error = statistics.stdev(max_accuracy_per_question) # can talk about switching to 'mean_accuracy_per_question'
    uncertainty_error = statistics.stdev(max_uncertainty_per_question) # can talk about switching to 'mean_uncertainty_per_question'
    
    
    popularity = math.log2(statistics.mean(list(df['pop'])))
    
    return (popularity, variety, mean_max_uncertainty, mean_max_accuracy, 
            accuracy_error, uncertainty_error)

