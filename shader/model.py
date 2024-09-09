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

class ParallelMLPWithPE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, hidden_dim2=16, output_dim=3, num_freqs=3, num_mlps=5000):
        super(ParallelMLPWithPE, self).__init__()
        self.num_mlps = num_mlps
        self.fc1_weight = nn.Parameter(torch.randn(num_mlps, input_dim + input_dim*num_freqs*2, hidden_dim))
        self.fc1_bias = nn.Parameter(torch.randn(num_mlps, hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(num_mlps, hidden_dim, hidden_dim2))
        self.fc2_bias = nn.Parameter(torch.randn(num_mlps, hidden_dim2))
        self.fc3_weight = nn.Parameter(torch.randn(num_mlps, hidden_dim2, output_dim))
        self.fc3_bias = nn.Parameter(torch.randn(num_mlps, output_dim))
        # initialize weights and biases
        nn.init.xavier_normal_(self.fc1_weight)
        nn.init.xavier_normal_(self.fc2_weight)
        nn.init.xavier_normal_(self.fc3_weight)
        nn.init.zeros_(self.fc1_bias)
        nn.init.zeros_(self.fc2_bias)
        nn.init.zeros_(self.fc3_bias)
        self.pe = PositionalEncoding(num_freqs)

    def forward(self, x, mlp_indices):
        x = self.pe(x)
        
        # Select weights and biases based on mlp_indices
        fc1_weight = self.fc1_weight[mlp_indices]
        fc1_bias = self.fc1_bias[mlp_indices]
        fc2_weight = self.fc2_weight[mlp_indices]
        fc2_bias = self.fc2_bias[mlp_indices]
        fc3_weight = self.fc3_weight[mlp_indices]
        fc3_bias = self.fc3_bias[mlp_indices]

        # Perform forward pass
        x = torch.bmm(x.unsqueeze(1), fc1_weight) + fc1_bias.unsqueeze(1)
        x = torch.bmm(x, fc2_weight) + fc2_bias.unsqueeze(1)
        x = torch.bmm(x, fc3_weight) + fc3_bias.unsqueeze(1)
        
        return torch.sigmoid(x).squeeze(1)
    
if __name__ == "__main__":
# 创建一个 ParallelMLPWithPE 实例
    model = ParallelMLPWithPE(input_dim=2, hidden_dim=64, hidden_dim2=16, output_dim=3, num_freqs=3, num_mlps=100)

    # 创建一个测试输入
    test_input = torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.3, 0.7], [0.3, 0.7]], dtype=torch.float32)
    mlp_indices = torch.tensor([0, 1, 2,68], dtype=torch.long)

    # 运行模型
    output = model(test_input, mlp_indices)

    # 打印输出
    print("输入:")
    print(test_input.shape)

    print("输出:")
    print(output.shape)