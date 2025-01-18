import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import kaleido

number_of_batches = 5
batch_size = 50

number_of_layers = {
    "meta-llama-Llama-2-7b-chat-hf": 32,
    "meta-llama-Llama-2-7b-hf": 32,
    "meta-llama-Llama-2-13b-chat-hf": 40,
    "meta-llama-Llama-2-13b-hf": 40,
    "meta-llama-Llama-2-70b-hf": 80
}

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
    
    with open(path_to_results + '/' + model + '/' + 'prob_question.pickle', 'rb') as handle:
        prob_question  = pickle.load(handle)
    with open(path_to_results + '/' + model + '/' + 'storer.pickle', 'rb') as handle:
        storer  = pickle.load(handle)

    similarity_storer = {}
    similarity_storer[current_prop] = {}

    for batch_index in tqdm(list(storer[current_prop].keys())):
        similarity_storer[current_prop][batch_index] = {}
        for ques_index in list(storer[current_prop][batch_index]):
            layer_wise_similarity = [0 for _ in range(number_of_layers[model])]
            
            for layer_index in range(number_of_layers[model]):
                layer_wise_vectors = []
                for paraphrase_index in range(len(storer[current_prop][batch_index][ques_index])):
                    layer_wise_vectors.append(storer[current_prop][batch_index][ques_index][paraphrase_index][layer_index])
                layer_wise_similarity[layer_index] = get_similarity(layer_wise_vectors, "cosine")
            similarity_storer[current_prop][batch_index][ques_index] = layer_wise_similarity
            
            
    fig = go.Figure()
    # cmap = plt.cm.get_cmap('viridis')
    cmap = matplotlib.colormaps['viridis']
    for batch_index in range(number_of_batches):
        rgba_color = cmap((batch_index+1) / number_of_batches)
        rgb_color = f'rgba({rgba_color[0]*255}, {rgba_color[1]*255}, {rgba_color[2]*255}, {rgba_color[3]})'
        batch_wise_similarity = np.zeros((batch_size, number_of_layers[model]), dtype = 'float32')
        batch_wise_prob = np.zeros((batch_size, number_of_layers[model]), dtype = 'float32')
        
        for ques_index in range(len(list(prob_question[current_prop][batch_index].keys()))):
            ques = list(prob_question[current_prop][batch_index].keys())[ques_index]
            for layer_index in range(number_of_layers[model]):
                batch_wise_similarity[ques_index][layer_index] = similarity_storer[current_prop][batch_index][ques][layer_index]
            batch_wise_prob[ques_index] = np.mean(np.array(prob_question[current_prop][batch_index][ques]), axis = 0)[1:]

        x = [layer_index for layer_index in range(number_of_layers[model])]
        y = np.mean(batch_wise_similarity, axis = 0)
        z = np.mean(batch_wise_prob, axis = 0)
        fig.add_trace(
            go.Scatter3d(x = x, y = y, z = z, 
                        name = 'Batch ' + str(batch_index),
                        mode = 'lines',
                        line=dict(color=rgb_color, width=5), showlegend=False)
        )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.75, y=-0.75, z=1.25)
    )
    fig.update_layout(
    scene_camera = camera,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        xaxis=dict(
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            tickfont=dict(size=14) 
        ),
        zaxis=dict(
            tickfont=dict(size=14)  
        )
    )
)
    # fig.update_scenes(xaxis_title_text='Layer')
    # fig.update_scenes(yaxis_title_text='Similarity')
    # fig.update_scenes(zaxis_title_text='Probability')


    if(os.path.exists(path_to_save + '/' + model + '/') == False):
        os.makedirs(path_to_save + '/' + model + '/')
    fig.write_html(path_to_save + '/' + model + '/' + current_prop + ".html")
    fig.write_image(path_to_save + '/' + model + '/' + current_prop + ".svg")