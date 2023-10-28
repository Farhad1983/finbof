import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.WQ = nn.Linear(input_dim, input_dim)
        self.WK = nn.Linear(input_dim, input_dim)
        self.WV = nn.Linear(input_dim, input_dim)
        self.WO = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Split the input into multiple heads
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        x = torch.matmul(attention_weights, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, input_dim)

        x = self.WO(x)

        return x, attention_weights

class TABL(nn.Module):
    def __init__(self, input_dim, output_dim,
                 projection_regularizer=None,
                 projection_constraint=None,
                 attention_regularizer=None,
                 attention_constraint=None,
                 num_heads=4):
        super(TABL, self).__init__()

        self.output_dim = output_dim
        self.projection_regularizer = projection_regularizer
        self.projection_constraint = projection_constraint
        self.attention_regularizer = attention_regularizer
        self.attention_constraint = attention_constraint

        self.attention = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)
        self.W1 = nn.Linear(input_dim, output_dim[0])
        self.W2 = nn.Linear(input_dim, output_dim[1])
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, output_dim[0], output_dim[1]), requires_grad=True)

    def forward(self, x):
        # Apply multi-head attention
        x, attention_weights = self.attention(x)

        # First mode projection
        x1 = self.W1(x)

        # Calculate attention
        W = torch.eye(x.size(1), dtype=x.dtype, device=x.device) + torch.ones(x.size(1), dtype=x.dtype, device=x.device) / x.size(1)
        attention = F.softmax(torch.matmul(x, W), dim=-1)

        # Apply attention
        x2 = self.alpha * x + (1.0 - self.alpha) * x * attention

        # Second mode projection
        x2 = self.W2(x2)

        # Bias add
        x = x1 + x2 + self.bias

        if self.output_dim[1] == 1:
            x = x.squeeze(dim=-1)

        return x, attention_weights

class TABNet(nn.Module):
    def __init__(self, input_dim, tabl_output_dim, num_classes, num_heads=4):
        super(TABNet, self).__init__()

        self.tabl_output_dim = tabl_output_dim
        self.tabl = TABL(input_dim=input_dim, output_dim=tabl_output_dim, num_heads=num_heads)

        self.fc = nn.Linear(tabl_output_dim[0], num_classes)

    def forward(self, x):
        x, _ = self.tabl(x)
        x = self.fc(x)
        return x