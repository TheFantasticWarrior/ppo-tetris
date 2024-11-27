import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, ff_size=1024, d_model=512):
        super(Model, self).__init__()
        self.d_model = d_model
        self.loc_embedding = nn.Linear(3, d_model)
        self.piece_embedding = nn.Embedding(8, d_model)
        self.rotation_embedding = nn.Linear(d_model, d_model*4)

        self.garbage_embedding = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.garbage_embedding2 = nn.Linear(64, ff_size)
        self.q_linear = nn.Linear(ff_size, 15 * d_model)
        self.k_linear = nn.Linear(d_model, d_model//4)
        self.v_linear = nn.Linear(d_model, d_model//4)
        self.qkv_linear = nn.Linear(15 * d_model, ff_size)
        self.attn_bias = Bias_Layer(ff_size)

        self.queue_linear = nn.Linear(ff_size, ff_size)
        self.remain_linear = nn.Linear(ff_size, ff_size)
        self.attn_norm = nn.LayerNorm(ff_size)
        self.ff_norm = nn.LayerNorm(ff_size)

        self.CNN = SimpleCNN(d_model, ff_size)

        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d)\
                    or isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, np.sqrt(2))

    def forward(self, x, retain=None):
        xyrot, queue, remains, board, garbage = x
        loc_processed = self.loc_embedding(xyrot)
        batch_size = queue.size(0)
        remain_indices = torch.nonzero(remains)+1
        if torch.any(remain_indices):
            pieces = torch.cat((queue, remain_indices), 1)
        else:
            pieces = queue
        pieces_processed = self.piece_embedding(pieces)
        pieces_processed[:, 0] += loc_processed
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).reshape(batch_size, -1, 4, self.d_model)
        pieces_visible = pieces_with_rotations[:, :7]
        pieces_not_visible = pieces_with_rotations[:, 7:]

        garbage_processed = self.garbage_embedding(
            garbage.unsqueeze(-1).float())
        garbage_processed = self.garbage_embedding2(
            garbage_processed.amax(-1)
        )

        board_processed = self.CNN(board.float().unsqueeze(1))

        state = board_processed+garbage_processed
        if retain is not None:
            state = state+retain

        def attn(state, pieces):
            num_heads = 8
            d_head = self.d_model // num_heads
            # Linear transformations
            q = self.q_linear(state).view(batch_size, 15,
                                          num_heads, d_head)
            k = self.k_linear(pieces_visible[:, i]).view(
                batch_size, -1, num_heads, d_head)
            v = self.v_linear(pieces_visible[:, i]).view(
                batch_size, -1, num_heads, d_head)

            bias = self.attn_bias(state).view(batch_size, num_heads, 15, 15)
            # Scaled dot-product attention
            attn_weights = torch.einsum(
                'bqnd,bknd->bnqk', q, k) / (d_head ** 0.5) + bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.einsum('bnqk,bvnd->bqnd', attn_weights, v)

            # Combine heads
            output = output.view(batch_size, 15 * self.d_model)
            return self.qkv_linear(output)

        for i in range(7):
            new_state = attn(state, pieces_visible[:, i:i+1])
            state = self.attn_norm(state+new_state)

            new_state = self.queue_linear(state)
            state = self.ff_norm(state+new_state)
        new_state = attn(state, pieces_not_visible)
        state = self.attn_norm(state+new_state)

        new_state = self.remain_linear(state)
        state = self.ff_norm(state+new_state)
        return state


class FinalModel(nn.Module):
    def __init__(self, ff_size=1024, d_model=512):
        super(FinalModel, self).__init__()
        self.d_model = d_model
        self.combine = nn.Linear(ff_size*2, d_model*10)
        self.t_encoder = nn.Transformer(
            d_model=d_model, nhead=8, dim_feedforward=ff_size,
            batch_first=True, num_encoder_layers=3, num_decoder_layers=0
        ).encoder
        self.linear = nn.Linear(d_model*10, ff_size)
        self.policy = nn.Linear(ff_size, 10)
        self.value = nn.Linear(ff_size, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        ins = torch.cat((x, y), -1)
        x = self.combine(ins).reshape(batch_size, 10, self.d_model)
        x = self.t_encoder(x).view(batch_size, -1)
        x = self.linear(x)
        logits = self.policy(x)
        value = self.value(x)
        return logits, value


class Bias_Layer(nn.Module):
    def __init__(self, ff_size):
        super(Bias_Layer, self).__init__()
        self.relu6 = nn.ReLU6()
        self.linear1 = nn.Linear(ff_size, 32, bias=False)
        self.linear2 = nn.Linear(32, 256)
        self.layernorm1 = nn.LayerNorm(256)
        self.linear3 = nn.Linear(256, 256*8)
        self.layernorm2 = nn.LayerNorm(256)
        self.linear4 = nn.Linear(256, 15*15)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu6(x)

        x = self.linear2(x)
        x = self.layernorm1(x)
        x = self.relu6(x)

        x = self.linear3(x).view(-1, 8, 256)
        x = self.layernorm2(x)
        x = self.relu6(x)

        x = self.linear4(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, d_model, out_size):
        super(SimpleCNN, self).__init__()

        # First layer: Depthwise separable convolution
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3)
        self.pointwise_conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU6()

        # Second layer: Depthwise separable convolution
        self.depthwise_conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                                         kernel_size=3, padding=1, groups=64)
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=64, out_channels=d_model, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(d_model)
        self.relu6 = nn.ReLU6()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(15*d_model, out_size)

    def forward(self, x):
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.pool(x)

        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn2(x)
        x = self.relu6(x)
        # Pad only at the bottom
        x = F.pad(x, (0, 1, 0, 0))
        x = self.pool(x)

        x = x.flatten(1)
        x = self.linear(x)
        return x
