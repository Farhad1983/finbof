import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalTemporalCorrelationBoFAdaptivePyramid(nn.Module):

    def __init__(self, window=50, split_horizon=5, n_codewords=256, n_conv=256, use_scaling=True):
        super(ConvolutionalTemporalCorrelationBoFAdaptivePyramid, self).__init__()

        self.split_horizon = split_horizon
        self.window = window
        self.n_codewords = n_codewords
        self.n_levels = int(window / split_horizon)
        self.use_scaling = use_scaling

        self.fc1 = nn.Linear(n_codewords * self.n_levels + n_codewords, 512)
        self.fc2 = nn.Linear(512, 3)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))

        self.a2 = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c2 = nn.Parameter(torch.FloatTensor(data=[0]))

        self.n1 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n2 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        self.n12 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n22 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        # Dictionary
        self.bof_conv = nn.Conv1d(n_conv, n_codewords, kernel_size=1)

        # Dictionary 2
        self.bof_conv2 = nn.Conv1d(144, n_codewords, kernel_size=1)

        # Input Convolutional
        self.input_conv = nn.Conv1d(144, n_conv, kernel_size=5, padding=2)
        self.bn_cnv = nn.BatchNorm1d(n_conv)

    def apply_temporal_bof_input(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv2(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = F.tanh(self.a2.expand_as(x)*x + self.c2.expand_as(x))
        else:
            x = F.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n12
        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, 15) * 15

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x


    def apply_temporal_bof(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = F.tanh(self.a.expand_as(x)*x + self.c.expand_as(x))
        else:
            x = F.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n1

        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, self.split_horizon) * self.n2

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x

    def forward(self, x):

        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        histogram2 = self.apply_temporal_bof_input(x)*0


        # Apply a convolutional layer
        x = self.input_conv(x)
        x = F.tanh(x)

        # Apply a temporal BoF
        temporal_histogram = self.apply_temporal_bof(x)

        temporal_histogram = torch.cat([temporal_histogram, histogram2], dim=1)

        # Classifier
        x = F.relu(self.fc1(temporal_histogram))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x
    

class ML_ConvolutionalTemporalCorrelationBoFAdaptivePyramid(nn.Module):

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

    def __init__(self, window=50, split_horizon=5, n_codewords=256, n_conv=256, use_scaling=True):
        super(ML_ConvolutionalTemporalCorrelationBoFAdaptivePyramid, self).__init__()

        self.split_horizon = split_horizon
        self.window = window
        self.n_codewords = n_codewords
        self.n_levels = int(window / split_horizon)
        self.use_scaling = use_scaling

        self.fc1 = nn.Linear(n_codewords * self.n_levels + n_codewords, 512)
        self.fc2 = nn.Linear(512, 3)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))

        self.a2 = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c2 = nn.Parameter(torch.FloatTensor(data=[0]))

        self.n1 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n2 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        self.n12 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n22 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        # Dictionary
        self.bof_conv = nn.Conv1d(n_conv, n_codewords, kernel_size=1)

        # Dictionary 2
        self.bof_conv2 = nn.Conv1d(144, n_codewords, kernel_size=1)

        # Input Convolutional
        self.input_conv = nn.Conv1d(144, n_conv, kernel_size=5, padding=2)
        self.bn_cnv = nn.BatchNorm1d(n_conv)

        self.timeDepthAttentions = [3,5,10,15]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        #self.timeDepthAttentions = nn.ParameterList(paramList)
        self.attentionDepth = nn.ParameterDict()
        self.attentionRate = nn.ParameterDict()

        self.attentionDepth1 = nn.ParameterDict()
        self.attentionRate1 = nn.ParameterDict()

        self.attentionDepth, self.attentionRate = self.getBreakDowns(15, self.timeDepthAttentions, 144)
        self.attentionDepth1, self.attentionRate1 = self.getBreakDowns1(15, self.timeDepthAttentions, n_conv)

    def apply_temporal_bof_input(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv2(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = F.tanh(self.a2.expand_as(x)*x + self.c2.expand_as(x))
        else:
            x = F.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n12
        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, 15) * 15

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x


    def apply_temporal_bof(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = F.tanh(self.a.expand_as(x)*x + self.c.expand_as(x))
        else:
            x = F.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n1

        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, self.split_horizon) * self.n2

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x

    def forward(self, x):
        ln = len(x)
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        



        XX = dict([])

        step_counter = 0
        for i in self.timeDepthAttentions:
            step_counter = step_counter + 1
            #print('attentionDepth Shape is i = ', i, ' :',self.attentionDepth[str(i)].repeat([ln ,1,1]).transpose(1, 2).shape)
            #print('x Shape is i = ', i, ' :',x.shape)
            #print('x Shape is 2 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1))
            #print('x Shape is 3 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1).shape)
            XX[i] = torch.mul(self.attentionDepth[str(i)].repeat([ln ,1,1]).transpose(1, 2).to(self.device), x.to(self.device))
            
            histogram2 = self.apply_temporal_bof_input(XX[i])*0

            # Apply a convolutional layer
            XX[i] = self.input_conv(XX[i])
            XX[i] = F.tanh(XX[i])

            # Apply a temporal BoF
            XX[i] = self.apply_temporal_bof(XX[i])

            #print('Shape xi: ', XX[i].shape)

            XX[i] = torch.cat([XX[i], histogram2], dim=1)

            #print('Shape xi: ', XX[i].shape)
            #print('Shape xi: ', self.attentionRate1[str(i)].repeat([ln ,1,1]).transpose(1, 2).shape)

            att_rate = torch.tensor(1/(len(self.timeDepthAttentions) - step_counter + 1)).repeat([ln ,1])
            att_rate = att_rate.expand_as(XX[i])
            #print('Shape att_rate: ', att_rate.shape)
            #print('Shape att_rate: ', att_rate)

            XX[i] = torch.mul(att_rate.to(self.device), XX[i].to(self.device))

        x= sum(XX[d] for d in self.timeDepthAttentions).div(len(self.timeDepthAttentions)).to(self.device)

        #x = torch.cat([x, histogram2], dim=1)
        

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x
