import os
from models.bof_models import ConvolutionalTemporalCorrelationBoFAdaptivePyramid, ML_ConvolutionalTemporalCorrelationBoFAdaptivePyramid
from lob_utils.train_anchored_utils import train_evaluate_anchored, get_average_metrics
from time import time
import pickle
from models.nn_models import  GRU_NN, LSTM_NN, CNN_NN, ML_LSTM_NN, ML_CNN_NN
from models.tabl_model import TABL_Layer, BTABL, CTABL, BL_layer
from models.BL import  BL_layer
from os.path import join
import torch
 

def run_experiments(model, output_path, train_epochs=20, window=10):

    a = time()
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print( 'device: ', device)
    results1 = train_evaluate_anchored(model, window=window, batch_size= 128, train_epochs=train_epochs, horizon=0, splits=range(9), device = device)
    b = time()

    print("----------")
    print("Elapsed time = ", b - a)
    metrics_1 = get_average_metrics(results1)

    print("----------")

    with open(join("results", output_path), 'wb') as f:
        pickle.dump([metrics_1, metrics_1, metrics_1], f)
        pickle.dump([results1, results1, results1], f)

    #os.system("shutdown /s /t 1")


# Experiments!

#model = lambda: ConvolutionalTemporalCorrelationBoFAdaptivePyramid(window=15, split_horizon=5, use_scaling=True)
#run_experiments(model, 'final_bof.pickle', window=15)

#model = lambda: LSTM_NN()
#run_experiments(model, 'final_MLLSTM.pickle', window=15)

#model = lambda: LSTM_NN()
#run_experiments(model, 'final_lstm.pickle', window=15)

#model = lambda: CNN_NN()
#run_experiments(model, 'final_cnn.pickle', window=15)

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)



