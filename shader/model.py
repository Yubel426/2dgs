import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=3, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(0, num_freqs) * torch.pi

    def forward(self, x):
        if self.include_input:
            encoded = [x]
        else:
            encoded = []
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)

class MLPWithPE(nn.Module):
    def __init__(self, index=0, input_dim=2, hidden_dim=64, hidden_dim2=16, output_dim=3, num_freqs=3):
        super(MLPWithPE, self).__init__()
        self.index = index
        self.fc1 = nn.Linear(input_dim + input_dim*num_freqs*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.pe = PositionalEncoding(num_freqs)

    def forward(self, x):
        x = self.pe(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.sigmoid(x)

