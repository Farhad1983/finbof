import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nn_models import  GRU_NN, LSTM_NN, CNN_NN

class MH_NN(nn.Module):

    def __init__(self, baseModel, timeDepthAttentions, window=15, split_horizon=5):
        super(MH_NN, self).__init__()

        self.baseModel = lambda: GRU_NN()
        self.timeDepthAttentions = timeDepthAttentions
        self.split_horizon = split_horizon
        self.window = window




    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        total, attentionDepth, attentionRate = getBreakDowns(self.window,self.timeDepthAttentions)
        XX = dict([])
        for i in self.timeDepthAttentions:
            XX[i] = torch.mul(attentionDepth[i], x)
            XX[i] = self.baseModel(XX[i])
            XX[i] = torch.mul(attentionRate[i], x)

        x= sum(XX[d] for d in self.timeDepthAttentions)

        return x
    
    def getBreakDowns(window_size, steps, future_dimention = 144):
        if not window_size in steps: steps.append(window_size)
        attentionDepth = dict([])
        attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            p1 = torch.ones(i,future_dimention)
            p2 = torch.zeros(window_size-i,future_dimention)
            p3 = torch.cat((p1, p2))
            attentionDepth[i] = p3
            result = result + p3

        for i in steps:
            attentionRate[i] = attentionDepth[i].div(result)
        #print(result)
        return result, attentionDepth, attentionRate
    