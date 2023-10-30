from os.path import join, exists
import pickle
import numpy as np
import pandas as pd
from lob_utils.train_anchored_utils import  get_average_metrics

def print_line(results):
    metrics = get_average_metrics(results)
    acc, precision, recall, f1, kappa = metrics

    print("$ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f $ & $ %5.4f \\pm %5.4f $"
          % (100*np.mean(acc), 100*np.std(acc),
             100*np.mean(precision), 100*np.std(precision),
             100 * np.mean(recall), 100 * np.std(recall),
             100 * np.mean(f1), 100 * np.std(f1),
             np.mean(kappa), np.std(kappa)))

def print_results(results_path):
    with open(join("results", results_path), 'rb') as f:
        [metrics_1, metrics_2, metrics_3] = pickle.load(f)
        [results1, results2, results3] = pickle.load(f)

    print("--------")
    print(results_path)
    print_line(results1)
    print(metrics_1)
    print(metrics_2)
    print(metrics_3)
    # print_line(results2)
    # print_line(results3)

def print_results_df(printOrginal = False, model_name = '', windowSize = 15):
    
    model_names = ['final_cnn','ML_CNN_NN','final_lstm','final_MLLSTM','final_bof','final_MLbof']
    bof15 = ['final_bof','final_MLbof']
    bof50 = ['final_bof','final_MLbof']
    lstm = ['final_lstm','final_MLLSTM']
    lstm50 = ['final_lstm_50','final_MLLSTM_50']
    cnn = ['final_cnn','ML_CNN_NN']
    cnn50 = ['final_cnn_50','ML_CNN_NN_50']


    if(model_name == 'bof'):
      model_names = bof15 if windowSize == 15 else bof50

    if(model_name == 'lstm'):
      model_names = lstm if windowSize == 15 else lstm50

    if(model_name == 'cnn'):
      model_names = cnn if windowSize == 15 else cnn50 

    metrics = ['acc_mean','acc_std','precision_mean','precision_std','recall_mean','recall_std','f1_mean','f1_std','kappa_mean','kappa_std']
    df = pd.DataFrame(columns=metrics, index = model_names)    
     
    for model in model_names:
        path = "{}.pickle".format(model)
        if(printOrginal):
          path = "orginal/{}.pickle".format(model)
        if (exists(join("results", path))):
          with open(join("results", path), 'rb') as f:
              [metrics_1, metrics_2, metrics_3] = pickle.load(f)
              [results1, results2, results3] = pickle.load(f)
              acc, precision, recall, f1, kappa = get_average_metrics(results1)
              df['acc_mean'][model] = np.mean(acc)
              df['precision_mean'][model] = np.mean(precision)
              df['recall_mean'][model] = np.mean(recall)
              df['f1_mean'][model] = np.mean(f1)
              df['kappa_mean'][model] = np.mean(kappa)
              
              df['acc_std'][model] = np.std(acc)
              df['precision_std'][model] = np.std(precision)
              df['recall_std'][model] = np.std(recall)
              df['f1_std'][model] = np.std(f1)
              df['kappa_std'][model] = np.std(kappa) 
    return df #.style.highlight_max( subset= ['acc_mean','acc_std','precision_mean','precision_std','recall_mean','recall_std'], color = 'lightgreen', axis = 0).highlight_max(subset= ['f1_mean','f1_std','kappa_mean','kappa_std'], color = 'yellow', axis = 0)
 


