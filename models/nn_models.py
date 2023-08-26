import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_NN(nn.Module):

    def __init__(self, n_gru_hidden=256):
        super(GRU_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)

        self.gru = nn.GRU(input_size=144, hidden_size=n_gru_hidden, num_layers=1)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(0, 1)

        h0 = torch.autograd.Variable(torch.zeros(1, x.size(1), self.n_gru_hidden).cuda())
        x = self.gru(x, h0)

        # Keep the last output
        x = x[-1].squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class LSTM_NN(nn.Module):

    def __init__(self, n_gru_hidden=256):
        super(LSTM_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)

        self.lstm = nn.LSTM(input_size=144, hidden_size=n_gru_hidden, num_layers=1)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(0, 1)

        # h0 = torch.autograd.Variable(torch.zeros(1, x.size(1), self.n_gru_hidden).cuda())
        x, (hn, cn) = self.lstm(x)

        # Keep the last output
        x = x[-1].squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class CNN_NN(nn.Module):

    def __init__(self, pooling=5, n_conv=256):
        super(CNN_NN, self).__init__()

        self.fc1 = nn.Linear(n_conv, 512)
        self.fc2 = nn.Linear(512, 3)

        # Input Convolutional
        self.input_conv = nn.Conv1d(144, n_conv, kernel_size=5, padding=2)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        # Apply a convolutional layer
        x = self.input_conv(x)
        x = F.tanh(x)

        x = F.avg_pool1d(x, x.size(2)).squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class ML_LSTM_NN(nn.Module):

    def __init__(self, n_gru_hidden=144):
        super(ML_LSTM_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)
        self.timeDepthAttentions = [3,5,10,15]
        self.window_size = 15
        self.batch_size = 128

        self.lstm = nn.LSTM(input_size=144, hidden_size=n_gru_hidden, num_layers=1)

        #print('window size: ',self.window_size)
        #print('timeDepthAttentions : ',self.timeDepthAttentions)
        
    def getBreakDowns(self, window_size, steps , future_dimention = 144):
        #print('steps : ',steps)
        #print('window_size : ',window_size)
        #print('future_dimention : ',future_dimention)
        if not window_size in steps : steps.append(window_size)
        attentionDepth = dict([])
        attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            #print('i value is: ',i)
            p1 = torch.ones( i,future_dimention)
            p2 = torch.zeros( window_size-i,future_dimention)
            #print('p1 shape: ', p1.shape)
            #print('p2 shape: ', p2.shape)
            p3 = torch.cat((p1, p2))
            attentionDepth[i] = p3
            result = result + p3 

        for i in steps:
            attentionRate[i] = attentionDepth[i].div(result)
        #print(result)
        return result, attentionDepth, attentionRate
    
    def forward(self, x):
        #print('X ** : ',x.shape) 
        x = x.squeeze()
        #print('X ** : ',x.shape) 
        #print('self.timeDepthAttentions ** : ',self.timeDepthAttentions)

        total, attentionDepth, attentionRate = self.getBreakDowns(self.window_size, self.timeDepthAttentions)
        XX = dict([])
        for i in self.timeDepthAttentions:
            #print('for i in: ',i)
            #print('attentionRate 4 i : ',attentionRate[i])
            #print('attentionRate 5 i : ',attentionRate[i].cuda().shape)
            #print('attentionRate 5 i : ',x.cuda().shape)
            XX[i] = torch.mul(attentionDepth[i].cuda(), x.cuda())
            #XX[i] = XX[i].transpose(0, 1)
            XX[i], (hn, cn) = self.lstm(XX[i])
            #print('XX 1 i : ',XX[i])
            #XX[i] = XX[i][-1].squeeze()
            #print('XX 2 i : ',XX[i])
            #print('XX 3 i : ',XX[i].shape)
             

            XX[i] = torch.mul(attentionRate[i].cuda(), XX[i].cuda())

        x= sum(XX[d] for d in self.timeDepthAttentions)
        x = x.unsqueeze(0)
        #print('X54 ** : ',x.shape)
        x = x[-1].squeeze()
        x = x[-1].squeeze()
        #print('X55 ** : ',x.shape)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        #print('X56 ** : ',x.shape)
        #print('X56 ** : ',x) 
        return x