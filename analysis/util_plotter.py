
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
from numpy import dot
from numpy.linalg import norm
from scipy import stats

# num_hidden_layers = model.config.num_hidden_layers
num_hidden_layers = 80
def get_distance(a, b, metric):
    a = np.array(a)
    b = np.array(b)
    if(metric == 'euclidean'): return norm(a-b)
    if(metric == 'cosine'): return dot(a, b)/(norm(a) * norm(b))
    
def ratio_diag(arr):
    
    # average of non-diagonal elements

    arr = arr.to_numpy()
    
    non_diag_mask = ~np.eye(arr.shape[0], dtype=bool)
    diag_mask = np.eye(arr.shape[0], dtype=bool)
    
    non_diagonal_elements = arr[non_diag_mask]
    diagonal_elements = arr[diag_mask]
    
    avg_non_diagonal = np.mean(non_diagonal_elements)
    avg_diagonal = np.mean(diagonal_elements)
    
    return avg_non_diagonal

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




def variety_plot(prop_wise_result, path_to_save, current_prop):
    
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
    plt.subplots_adjust(wspace=1, hspace=1)
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim([12, 20])
            ax[i, j].set_ylim([0, 0.6])
    fontsize = 14
    
    ax[0, 0].scatter(popularity, mean_max_accuracy, color='blue')
    ax[0, 0].set_ylabel('Mean Max Acc. (F1-Score) \nper question batch-wise', fontsize = fontsize)
    ax[0, 0].set_xlabel('log popularity', fontsize = fontsize)
    ax[0, 0].xaxis.grid(True)
    
    ax[0, 1].scatter(popularity, accuracy_error, color = 'blue', edgecolor=(0, 0, 0, 0.5), linewidths=1.5, s = 75)
    
    perform_statistical_test = False
    if(perform_statistical_test):
        print(len(popularity))
        print(len(accuracy_error))
        print(popularity)
        print(accuracy_error)
        res = stats.spearmanr(popularity, accuracy_error, alternative = 'greater')
        print(res.statistic, res.pvalue)
    # ax[0, 1].set_ylabel('Stdev Max Acc. (F1-Score) \nper question batch-wise', fontsize = fontsize)
    # ax[0, 1].set_xlabel('log popularity', fontsize = fontsize)
    ax[0, 1].xaxis.grid(True, color = 'white')
    ax[0, 1].set_facecolor((0.95, 0.95, 0.95))


    ax[1, 0].scatter(popularity, mean_max_uncertainty, color='blue')
    ax[1, 0].set_ylabel('Mean Max Uncertainty \nper question batch-wise', fontsize = fontsize)
    ax[1, 0].set_xlabel('log popularity', fontsize = fontsize)
    ax[1, 0].xaxis.grid(True)

    ax[1, 1].scatter(popularity, uncertainty_error, color = 'blue')
    ax[1, 1].set_ylabel('Stdev Max Uncertainty \nper question batch-wise', fontsize = fontsize)
    ax[1, 1].set_xlabel('log popularity', fontsize = fontsize)
    ax[1, 1].xaxis.grid(True)

    extent = ax[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path_to_save + current_prop + '_ax00.svg', bbox_inches=extent.expanded(1.4, 1.5), format = 'svg')
    extent = ax[0, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path_to_save + current_prop + '_ax01.svg', bbox_inches=extent.expanded(1.4, 1.5), format = 'svg')
    extent = ax[1, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path_to_save + current_prop + '_ax10.svg', bbox_inches=extent.expanded(1.4, 1.5), format = 'svg')
    extent = ax[1, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path_to_save + current_prop + '_ax11.svg', bbox_inches=extent.expanded(1.4, 1.5), format = 'svg')

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

