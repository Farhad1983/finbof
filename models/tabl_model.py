from models.BL import BL_layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class TABL_Layer(nn.Module):
    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        self.t1 = t1

        weight1 = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')
        
        weight = torch.Tensor(t1, t1)
        self.W = nn.Parameter(weight)
        nn.init.constant_(self.W, 1/t1)
 
        weight2 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1,)
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, X):
        
        #x = x.transpose(0, 1)
        #maintaining the weight parameter between 0 and 1.
        if (self.l[0] < 0): 
          l = torch.Tensor(1,)
          self.l = nn.Parameter(l)
          nn.init.constant_(self.l, 0.0)

        if (self.l[0] > 1): 
          l = torch.Tensor(1,)
          self.l = nn.Parameter(l)
          nn.init.constant_(self.l, 1.0)
     
        #modelling the dependence along the first mode of X while keeping the temporal order intact (7)
        X = self.W1 @ X

        #enforcing constant (1) on the diagonal
        W = self.W -self.W *torch.eye(self.t1,dtype=torch.float32).to(self.device )+torch.eye(self.t1,dtype=torch.float32).to(self.device )/self.t1

        #attention, the aim of the second step is to learn how important the temporal instances are to each other (8)
        E = X @ W

        #computing the attention mask  (9)
        A = torch.softmax(E, dim=-1)

        #applying a soft attention mechanism  (10)
        #he attention mask A obtained from the third step is used to zero out the effect of unimportant elements
        X = self.l[0] * (X) * A + (1.0 - self.l[0]) * X

        #the final step of the proposed layer estimates the temporal mapping W2, after the bias shift (11)
        y = X @ self.W2 + self.B
        return y
    

class BTABL(nn.Module):
  def __init__(self, d2, d1, t1, t2, d3, t3):
    super().__init__()

    self.BL = BL_layer(d2, d1, t1, t2)
    self.TABL = TABL_Layer(d3, d2, t2, t3)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):

    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    x = torch.squeeze(x)
    x = torch.softmax(x, 1)
    return x

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))



class CTABL(nn.Module):
  def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
    super().__init__()
    
    self.BL = BL_layer(d2, d1, t1, t2)
    self.BL2 = BL_layer(d3, d2, t2, t3)
    self.TABL = TABL_Layer(d4, d3, t3, t4)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    x = x.transpose(1, 2)
    #print('x:', x.shape)
    self.max_norm_(self.BL.W1.data)
    #print('x2:', x.shape)
    self.max_norm_(self.BL.W2.data)
    #print('x3:', x.shape)
    x = self.BL(x)
    #print('x4:', x.shape)
    x = self.dropout(x)

    self.max_norm_(self.BL2.W1.data)
    self.max_norm_(self.BL2.W2.data)
    x = self.BL2(x)
    x = self.dropout(x)

    #print('x5:', x.shape)
    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    #print('x6:', x.shape)
    
    x = torch.squeeze(x)
    x = torch.softmax(x, 1)
    return x
  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))


class MT_CTABL(nn.Module):
  def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
    super().__init__()
    
    self.CTABL1 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
    self.CTABL2 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
    self.CTABL3 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
    self.CTABL4 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
    self.CTABL5 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
    self.CTABL6 = CTABL(d2, d1, t1, t2, d3, t3, d4, t4)
   
    self.dropout = nn.Dropout(0.1)

    self.timeDepthAttentions = [3,5,10,15]
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #self.timeDepthAttentions = nn.ParameterList(paramList)
    self.attentionDepth = nn.ParameterDict()
    self.attentionRate = nn.ParameterDict()

    #self.attentionDepth1 = nn.ParameterDict()
    self.attentionRate1 = nn.ParameterDict()

    self.attentionDepth, self.attentionRate = self.getBreakDowns(15, self.timeDepthAttentions, 144)
    self.attentionRate1 = self.getBreakDowns1(15, self.timeDepthAttentions, 3)

  def forward(self, x):
    ln = len(x)
    #x = x.transpose(1, 2)


    #print('ln :', ln)

    XX = dict([])
    cntr = 0
    for i in self.timeDepthAttentions:
        #print('i: ', i)
        cntr = cntr + 1
        #print('cntr: ', cntr)
        #print('ln :', self.attentionDepth[str(i)].repeat([ln ,1,1]).shape)
        #print('ln :', x.shape)
        #print('x Shape is 1 : ',attentionDepth[i].repeat([ln ,1,1]).shape)
        #print('x Shape is 2 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1))
        #print('x Shape is 3 : ',attentionDepth[i].repeat([ln ,1,1]).transpose(0, 1).shape)
        XX[i] = torch.mul(self.attentionDepth[str(i)].repeat([ln ,1,1]).to(self.device), x.to(self.device))
        #XX[i], (hn, cn) = self.lstm(XX[i])

        if cntr == 1:
           XX[i] = self.CTABL1(XX[i])
        if cntr == 2:
           XX[i] = self.CTABL2(XX[i])
        if cntr == 3:
           XX[i] = self.CTABL3(XX[i])
        if cntr == 4:
           XX[i] = self.CTABL4(XX[i])
        if cntr == 5:
           XX[i] = self.CTABL5(XX[i])
        if cntr == 6:
           XX[i] = self.CTABL6(XX[i])
        #print('ln :', XX[i].shape)
        #print('ln :', self.attentionRate1[str(i)].repeat([ln ,1,1]).shape)

        XX[i] = torch.mul(self.attentionRate1[str(i)].to(self.device), XX[i].to(self.device))
        #print('X --> 0 ', XX[i].shape)
   
    x= sum(XX[d] for d in self.timeDepthAttentions).to(self.device)
    #print('X --> 10 ', x.shape)
    x = torch.squeeze(x)
    #print('X --> 20 ', x.shape)
    x = torch.softmax(x, 1)
    #print('X --> 30 ', x.shape)
    return x
  

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))

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
      
      total_sum = 0
      for i in steps:
         total_sum = total_sum + i
      for i in steps:
          weight2 = torch.ones(future_dimention) * i/total_sum
          self.attentionRate1[str(i)] = nn.Parameter(weight2)

      return  self.attentionRate1