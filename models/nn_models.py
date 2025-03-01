import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRU_NN(nn.Module):

    def __init__(self, n_gru_hidden=256):
        super(GRU_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)

        self.gru = nn.GRU(input_size=144, hidden_size=n_gru_hidden, num_layers=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(0, 1)
        h0 = torch.autograd.Variable(torch.zeros(1, x.size(1), self.n_gru_hidden).to(self.device))
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

    def getBreakDowns(self, window_size, steps , future_dimention = 144):
        if not str(window_size) in steps : steps.append(window_size)
        #attentionDepth = dict([])
        #attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            p1 = torch.ones( i,future_dimention)
            p2 = torch.zeros( window_size-i,future_dimention)
            p3 = torch.cat((p1, p2))
            self.attentionDepth[str(i)] = p3
            result = result + p3 

        for i in steps:
            self.attentionRate[str(i)] = self.attentionDepth[str(i)].div(result)
        #print(result)
        return self.attentionDepth, self.attentionRate

    def getBreakDowns1(self, window_size, steps , future_dimention = 144):
        if not str(window_size) in steps : steps.append(window_size)
        #attentionDepth = dict([])
        #attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            p1 = torch.ones( i,future_dimention)
            p2 = torch.zeros( window_size-i,future_dimention)
            p3 = torch.cat((p1, p2))
            self.attentionDepth1[str(i)] = p3
            result = result + p3 

        for i in steps:
            self.attentionRate1[str(i)] = self.attentionDepth1[str(i)].div(result)
        #print(result)
        return self.attentionDepth1, self.attentionRate1
    
    def __init__(self, n_gru_hidden=256):
        super(ML_LSTM_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)
        self.fc3 = nn.Linear(512, 3)
        self.timeDepthAttentions = [3,5,10,15]
        #self.window_size = Variable(15 )

        self.lstm = nn.LSTM(input_size=144, hidden_size=n_gru_hidden, num_layers=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.timeDepthAttentions = nn.ParameterList(paramList)
        self.attentionDepth = nn.ParameterDict()
        self.attentionRate = nn.ParameterDict()

        self.attentionDepth1 = nn.ParameterDict()
        self.attentionRate1 = nn.ParameterDict()

        self.attentionDepth, self.attentionRate = self.getBreakDowns(15, self.timeDepthAttentions, 144)
        self.attentionDepth1, self.attentionRate1 = self.getBreakDowns1(15, self.timeDepthAttentions, self.n_gru_hidden)

        #print('self.attentionDepth: ', self.attentionDepth)

        
    
    
    def forward(self, x):
        ln = len(x)
        x = x.transpose(0, 1)
        #timeDepthAttentions =  [3,5,10,15]
        #total, attentionDepth, attentionRate = self.getBreakDowns(15, timeDepthAttentions)
        #total1, attentionDepth1, attentionRate1 = self.getBreakDowns(15, timeDepthAttentions, self.n_gru_hidden)

        #attentionDepth = attentionDepth.to(self.device)
        #attentionRate = attentionRate.to(self.device)
        #total = total.to(self.device)
        #print('x Shape is : ',x.shape)

        XX = dict([])
        for i in self.timeDepthAttentions:
            #print('x Shape is 1 : ',attentionDepth[i].repeat([ln ,1,1]).shape)
            #print('x Shape is 2 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1))
            #print('x Shape is 3 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1).shape)
            XX[i] = torch.mul(self.attentionDepth[str(i)].repeat([ln ,1,1]).transpose(0, 1).to(self.device), x.to(self.device))
            XX[i], (hn, cn) = self.lstm(XX[i])
            XX[i] = torch.mul(self.attentionRate1[str(i)].repeat([ln ,1,1]).transpose(0, 1).to(self.device), XX[i].to(self.device))

        x= sum(XX[d] for d in self.timeDepthAttentions).to(self.device)
        #tsum = total.repeat([len(x),1,1]).to(self.device)
        #x = x.div(tsum).to(self.device)

        #print('self.attentionDepth[str(3)]: ',self.attentionDepth[str(3)])

        #x = x.unsqueeze(0)
       
        #x = x[-1].squeeze()
        
        x = x[-1].squeeze()
        #print('x Shape is : ',x.shape)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        
        #print('x Shape is : ',x.shape)

        return x
    


class ML_CNN_NN(nn.Module):

    def getBreakDowns(self, window_size, steps , future_dimention = 144):
        if not str(window_size) in steps : steps.append(window_size)
        #attentionDepth = dict([])
        #attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            p1 = torch.ones( i,future_dimention)
            p2 = torch.zeros( window_size-i,future_dimention)
            p3 = torch.cat((p1, p2))
            self.attentionDepth[str(i)] = p3
            result = result + p3 

        for i in steps:
            self.attentionRate[str(i)] = self.attentionDepth[str(i)].div(result)
        #print(result)
        return self.attentionDepth, self.attentionRate

    def getBreakDowns1(self, window_size, steps , future_dimention = 144):
        if not str(window_size) in steps : steps.append(window_size)
        #attentionDepth = dict([])
        #attentionRate = dict([])
        result = torch.zeros(window_size,future_dimention)
        for i in steps:
            p1 = torch.ones( i,future_dimention)
            p2 = torch.zeros( window_size-i,future_dimention)
            p3 = torch.cat((p1, p2))
            self.attentionDepth1[str(i)] = p3
            result = result + p3 

        for i in steps:
            self.attentionRate1[str(i)] = self.attentionDepth1[str(i)].div(result)
        #print(result)
        return self.attentionDepth1, self.attentionRate1
    
    def __init__(self, pooling=5, n_conv=256):
        super(ML_CNN_NN, self).__init__()

        self.fc1 = nn.Linear(n_conv, 512)
        self.fc2 = nn.Linear(512, 3)

        # Input Convolutional
        self.input_conv = nn.Conv1d(144, n_conv, kernel_size=5, padding=2)

        self.timeDepthAttentions = [3,5,10,15]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.timeDepthAttentions = nn.ParameterList(paramList)
        self.attentionDepth = nn.ParameterDict()
        self.attentionRate = nn.ParameterDict()

        self.attentionDepth1 = nn.ParameterDict()
        self.attentionRate1 = nn.ParameterDict()

        self.attentionDepth, self.attentionRate = self.getBreakDowns(15, self.timeDepthAttentions, 144)
        self.attentionDepth1, self.attentionRate1 = self.getBreakDowns1(15, self.timeDepthAttentions, n_conv)

        print('self.attentionDepth: ', self.attentionDepth)

    def forward(self, x):
        ln = len(x)
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        XX = dict([])
        for i in self.timeDepthAttentions:
            #print('x Shape is 1 : ',attentionDepth[i].repeat([ln ,1,1]).shape)
            #print('x Shape is 2 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1))
            #print('x Shape is 3 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1).shape)
            XX[i] = torch.mul(self.attentionDepth[str(i)].repeat([ln ,1,1]).transpose(1, 2).to(self.device), x.to(self.device))
            
            XX[i] = self.input_conv(XX[i])
            XX[i] = F.tanh(XX[i])

            XX[i] = torch.mul(self.attentionRate1[str(i)].repeat([ln ,1,1]).transpose(1, 2).to(self.device), XX[i].to(self.device))

        x= sum(XX[d] for d in self.timeDepthAttentions).to(self.device)

        # Apply a convolutional layer
        
        x = F.avg_pool1d(x, x.size(2)).squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x