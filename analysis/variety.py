import util_plotter
import pandas as pd
import os
import argparse

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--path_to_save", help="path of the directory to save output plots", required=True)
    parser.add_argument("-c", "--current_prop", help="name of the property", required=True)
    parser.add_argument("-i", "--path_to_results", help="path of the directory containing input data files", required=True)
    parser.add_argument("-m", "--model_name", help="name of the model", required=True)

    args = parser.parse_args()
    
    path_to_model = args.path_to_results + args.model_name 
    prop_wise_result = []
    for batch_file in os.listdir(path_to_model + '/' + args.current_prop):
        batch_wise_result = []
        df = pd.read_csv(path_to_model + '/' + args.current_prop + '/' + batch_file)
        
        features = list(df.columns)
        data_extracted = util_plotter.extract_results(df)
        
        for item in data_extracted:
            batch_wise_result.append(item)
        prop_wise_result.append(batch_wise_result)

    if not os.path.exists(args.path_to_save + args.model_name + '/'):
        os.makedirs(args.path_to_save + args.model_name + '/')
        
    util_plotter.variety_plot(prop_wise_result, args.path_to_save + args.model_name + '/' + args.current_prop + '.png')