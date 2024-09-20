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
    def __init__(
        self,
        index=0,
        input_dim=2,
        hidden_dim=64,
        hidden_dim2=16,
        output_dim=3,
        num_freqs=3,
    ):
        super(MLPWithPE, self).__init__()
        self.index = index
        self.fc1 = nn.Linear(input_dim + input_dim * num_freqs * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.pe = PositionalEncoding(num_freqs)

    def forward(self, x):
        x = self.pe(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


class ParallelMLPWithPE(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        hidden_dim2=16,
        output_dim=3,
        num_freqs=3,
        num_freqs_dir=2,
        num_mlps=5000,
    ):
        super(ParallelMLPWithPE, self).__init__()
        self.num_mlps = num_mlps
        self.fc1_weight = nn.Parameter(
            torch.randn(num_mlps, input_dim + input_dim * num_freqs * 2, hidden_dim)
        )
        # self.fc1_weight = nn.Parameter(
        #     torch.randn(num_mlps, input_dim + input_dim * num_freqs * 2 + num_freqs_dir * 6 + 3, hidden_dim)
        # )
        self.fc1_bias = nn.Parameter(torch.randn(num_mlps, hidden_dim))
        self.fc2_weight = nn.Parameter(torch.randn(num_mlps, hidden_dim, hidden_dim2))
        self.fc2_bias = nn.Parameter(torch.randn(num_mlps, hidden_dim2))
        self.fc3_weight = nn.Parameter(torch.randn(num_mlps, hidden_dim2 + num_freqs_dir * 6 + 3, output_dim))
        self.fc3_bias = nn.Parameter(torch.randn(num_mlps, output_dim))
        # initialize weights and biases
        nn.init.xavier_normal_(self.fc1_weight)
        nn.init.xavier_normal_(self.fc2_weight)
        nn.init.xavier_normal_(self.fc3_weight)
        nn.init.zeros_(self.fc1_bias)
        nn.init.zeros_(self.fc2_bias)
        nn.init.zeros_(self.fc3_bias)
        self.pe = PositionalEncoding(num_freqs)
        self.pe_dir = PositionalEncoding(num_freqs_dir)

    def forward(self, x, mlp_indices, dir=None):
        x = self.pe(x)
        if dir is not None:
            dir = self.pe_dir(dir)
            # x = torch.cat([x, dir], dim=-1)
        # Select weights and biases based on mlp_indices
        fc1_weight = self.fc1_weight[mlp_indices]
        fc1_bias = self.fc1_bias[mlp_indices]
        fc2_weight = self.fc2_weight[mlp_indices]
        fc2_bias = self.fc2_bias[mlp_indices]
        fc3_weight = self.fc3_weight[mlp_indices]
        fc3_bias = self.fc3_bias[mlp_indices]

        # Perform forward pass
        x = torch.bmm(x.unsqueeze(1), fc1_weight) + fc1_bias.unsqueeze(1)
        x = F.relu(x)
        x = torch.bmm(x, fc2_weight) + fc2_bias.unsqueeze(1)
        x = F.relu(x)
        x = torch.cat([x, dir.unsqueeze(1)], dim=-1)
        x = torch.bmm(x, fc3_weight) + fc3_bias.unsqueeze(1)
        return F.sigmoid(x).squeeze(1)

    def densify_and_prune(self, prune_mask):
        valid_points_mask = ~prune_mask
        self.fc1_weight = nn.Parameter(self.fc1_weight[valid_points_mask])
        self.fc1_bias = nn.Parameter(self.fc1_bias[valid_points_mask])
        self.fc2_weight = nn.Parameter(self.fc2_weight[valid_points_mask])
        self.fc2_bias = nn.Parameter(self.fc2_bias[valid_points_mask])
        self.fc3_weight = nn.Parameter(self.fc3_weight[valid_points_mask])
        self.fc3_bias = nn.Parameter(self.fc3_bias[valid_points_mask])
   
    def densify_and_split(self, mask):
        new_fc1_weight = self.fc1_weight[mask].repeat(2, 1, 1)
        new_fc1_bias = self.fc1_bias[mask].repeat(2, 1)
        new_fc2_weight = self.fc2_weight[mask].repeat(2, 1, 1)
        new_fc2_bias = self.fc2_bias[mask].repeat(2, 1)
        new_fc3_weight = self.fc3_weight[mask].repeat(2, 1, 1)
        new_fc3_bias = self.fc3_bias[mask].repeat(2, 1)
        mask = ~mask
        self.fc1_weight = nn.Parameter(torch.cat((self.fc1_weight[mask], new_fc1_weight), dim=0))
        self.fc1_bias = nn.Parameter(torch.cat((self.fc1_bias[mask], new_fc1_bias), dim=0))
        self.fc2_weight = nn.Parameter(torch.cat((self.fc2_weight[mask], new_fc2_weight), dim=0))
        self.fc2_bias = nn.Parameter(torch.cat((self.fc2_bias[mask], new_fc2_bias), dim=0))
        self.fc3_weight = nn.Parameter(torch.cat((self.fc3_weight[mask], new_fc3_weight), dim=0))
        self.fc3_bias = nn.Parameter(torch.cat((self.fc3_bias[mask], new_fc3_bias), dim=0))

    def densify_and_clone(self, mask):
        new_fc1_weight = self.fc1_weight[mask]
        new_fc1_bias = self.fc1_bias[mask]
        new_fc2_weight = self.fc2_weight[mask]
        new_fc2_bias = self.fc2_bias[mask]
        new_fc3_weight = self.fc3_weight[mask]
        new_fc3_bias = self.fc3_bias[mask]
        self.fc1_weight = nn.Parameter(torch.cat((self.fc1_weight, new_fc1_weight), dim=0))
        self.fc1_bias = nn.Parameter(torch.cat((self.fc1_bias, new_fc1_bias), dim=0))
        self.fc2_weight = nn.Parameter(torch.cat((self.fc2_weight, new_fc2_weight), dim=0))
        self.fc2_bias = nn.Parameter(torch.cat((self.fc2_bias, new_fc2_bias), dim=0))
        self.fc3_weight = nn.Parameter(torch.cat((self.fc3_weight, new_fc3_weight), dim=0))
        self.fc3_bias = nn.Parameter(torch.cat((self.fc3_bias, new_fc3_bias), dim=0))


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open(pth, "wb") as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)
        ).save(f, "PNG")


def test_parallel_mlp_with_pe(xy_coords):
    # 加载图片
    gt = (
        torch.tensor(np.array(Image.open("test.png")).transpose(2, 0, 1) / 255)
        .float()
        .cuda()
    )
    # 创建 ParallelMLPWithPE 实例
    model = ParallelMLPWithPE(
        input_dim=2,
        hidden_dim=64,
        hidden_dim2=16,
        output_dim=3,
        num_freqs=3,
        num_mlps=25600,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    index = (
        torch.range(0, 25599)
        .repeat_interleave(5)
        .reshape(160, 800)
        .repeat_interleave(5, dim=0)
        .reshape(-1)
        .long()
        .cuda()
    )
    if not os.path.exists("./test/ParallelMLPWithPE_pi"):
        os.mkdir("./test/ParallelMLPWithPE_pi")
    else:
        os.system("rm -rf ./test/ParallelMLPWithPE_pi/*")
    save_img_u8(
        gt.cpu().permute(1, 2, 0).detach().numpy(),
        os.path.join("./test/ParallelMLPWithPE_pi/image_gt.png"),
    )
    for i in range(1000):
        optimizer.zero_grad()
        rgb_output = model(xy_coords, index)
        img = rgb_output.reshape(3, 800, 800)
        loss = torch.abs((img - gt[0:3, :])).mean()
        loss.backward()
        optimizer.step()
        print("Loss:", loss.item(), "Step:", i)
        if i % 100 == 0:
            save_img_u8(
                img.cpu().permute(1, 2, 0).detach().numpy(),
                os.path.join(
                    "./test/ParallelMLPWithPE_pi/image_" + str(i) + "_" + ".png"
                ),
            )


def generate_local_coordinates(image_size=800, block_size=5):
    # 生成全局坐标
    width, height = image_size, image_size
    x = torch.linspace(0, width - 1, width)
    y = torch.linspace(0, height - 1, height)
    xv, yv = torch.meshgrid(x, y, indexing="ij")

    # 生成局部坐标
    local_x = torch.linspace(-(block_size // 2), block_size // 2, block_size)
    local_y = torch.linspace(-(block_size // 2), block_size // 2, block_size)
    local_xv, local_yv = torch.meshgrid(local_x, local_y, indexing="ij")
    local_coords = torch.stack([local_xv, local_yv], dim=-1)  # (5, 5, 2)

    # 将局部坐标扩展到整个图片
    local_coords_expanded = local_coords.repeat(
        width // block_size, height // block_size, 1
    )

    # # 计算每个小方块的起始位置
    # block_indices_x = torch.arange(0, width, block_size)
    # block_indices_y = torch.arange(0, height, block_size)
    # block_xv, block_yv = torch.meshgrid(block_indices_x, block_indices_y, indexing='ij')
    # block_start_coords = torch.stack([block_xv, block_yv], dim=-1).unsqueeze(2).unsqueeze(3)  # (160, 160, 1, 1, 2)

    # # 计算最终坐标
    # final_coords = block_start_coords + local_coords_expanded  # (160, 160, 5, 5, 2)
    # final_coords = final_coords.reshape(-1, 2)  # (25600 * 25, 2)

    return local_coords_expanded.reshape(-1, 2) / 2 + 0.5


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import os

    # 测试生成的坐标
    coords = generate_local_coordinates().cuda()
    print(coords.shape)
    test_parallel_mlp_with_pe(coords)
