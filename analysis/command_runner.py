import argparse
import subprocess
import os
import shutil 


models = ['meta-llama-Llama-2-7b-chat-hf', 'meta-llama-Llama-2-7b-hf', 'meta-llama-Llama-2-13b-chat-hf', 'meta-llama-Llama-2-13b-hf', 'meta-llama-Llama-2-70b-hf']
relations = ['capital', 'genre', 'occupation', 'religion', 'screenwriter', 'sport']

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode", required=True)
    args = parser.parse_args()
    mode = args.mode
    
    if(mode == 'variety'):
        for model in models[:-1]:
            for relation in relations:
                print("Currently running ... {} + {}".format(model, relation) + "...")
                result = subprocess.run('python3 variety.py -o /home/prasoon/snap/main/information-anxiety/plots/variety -c {} -i /home/prasoon/snap/main/information-anxiety/results/variety -m {}'.format(relation, model), shell=True)
                # uncomment to print the any terminal output for each 'neural.py' running
                # print(f"Output:\n{result.stdout}")
            
    
    if(mode == 'neural'):
        for model in models:
            for relation in relations:
                print("Currently running ... {} + {}".format(model, relation) + "...")
                result = subprocess.run('python3 neural.py -o /home/prasoon/snap/main/information-anxiety/plots/neural -c {} -i /home/prasoon/snap/main/information-anxiety/results/neural -m {}'.format(relation, model), shell=True, capture_output=True, text=True)
                # uncomment to print the any terminal output for each 'neural.py' running
                # print(f"Output:\n{result.stdout}")
    
    if(mode == 'attention'):
        for model in models:
            for relation in relations:
                print("Currently running ... {} + {}".format(model, relation) + "...")
                result = subprocess.run('python3 attention.py -o /home/prasoon/snap/main/information-anxiety/plots/attention -c {} -i /home/prasoon/snap/main/information-anxiety/results/attention -m {}'.format(relation, model), shell=True)
                # uncomment to print the any terminal output for each 'neural.py' running
                # print(f"Output:\n{result.stdout}")

    
    if(mode == 'distance-hidden'):
        for model in models:
            for relation in relations:
                print("Currently running ... {} + {}".format(model, relation) + "...")
                result = subprocess.run('python3 distance-hidden.py -o /home/prasoon/snap/main/information-anxiety/plots/distance-hidden -c {} -i /home/prasoon/snap/main/information-anxiety/results/distance-hidden -m {}'.format(relation, model), shell=True)
                # uncomment to print the any terminal output for each 'neural.py' running
                
    if(mode == 'matrix'):
        for model in models:
            print("Currently running ... {}".format(model) + "...")
            result = subprocess.run('python3 matrix.py -o /home/prasoon/snap/main/information-anxiety/plots/matrix -i /home/prasoon/snap/main/information-anxiety/results/matrix -m {}'.format(model), shell = True)
            # uncomment to print the any terminal output for each 'neural.py' running
            # print(f"Output:\n{result.stdout}")
        
    if(mode == 'matrixadd'):
        for model in models:
            print("Currently running ... {}".format(model) + "...")
            result = subprocess.run('python3 matrixadd.py -o /home/prasoon/snap/main/information-anxiety/plots/matrixadd -i /home/prasoon/snap/main/information-anxiety/results/matrixadd -m {}'.format(model), shell = True)
            # uncomment to print the any terminal output for each 'matrixadd.py' running
            # print(f"Output:\n{result.stdout}")    


    if(mode == 'misc'):
        path_to_dir = '/home/prasoon/snap/main/information-anxiety/plots/variety/'
        for model in models:
            for relation in relations:
                result = subprocess.run('rm '+path_to_dir+model+'/'+relation+'.png')
    
    if(mode == 'neuraladd'):
        for model in models:
            for relation in relations:
                print("Currently running ... {} + {}".format(model, relation) + "...")
                result = subprocess.run('python3 neuraladd.py -o /home/prasoon/snap/main/information-anxiety/plots/neuraladd -c {} -i /home/prasoon/snap/main/information-anxiety/results/neuraladd -m {}'.format(relation, model), shell=True)
