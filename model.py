import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, ff_size=1024, d_model=512):
        super(Model, self).__init__()
        self.d_model = d_model
        self.piece_embedding = nn.Embedding(8, d_model)
        self.rotation_embedding = nn.Linear(d_model, d_model*4)

        self.garbage_embedding = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.garbage_embedding2 = nn.Linear(64, ff_size)
        self.q_linear = nn.Linear(ff_size, 18 * d_model)
        self.k_linear = nn.Linear(d_model*4, d_model)
        self.v_linear = nn.Linear(d_model*4, d_model)
        self.attn_bias = Bias_Layer(d_model)

        self.queue_linear = nn.Linear(ff_size, ff_size)
        self.remian_linear = nn.Linear(ff_size, ff_size)
        self.layer_norm = nn.LayerNorm(d_model)

        self.CNN = SimpleCNN(d_model)
        self.retain = None

        self.policy = nn.Linear(ff_size, 10)
        self.value = nn.Linear(ff_size, 1)

    def forward(self, x):
        queue, remains, board, garbage = x
        batch_size = queue.size(0)
        remain_indices = torch.nonzero(remains)
        pieces = torch.cat((queue, remain_indices), 1)
        pieces_processed = self.piece_embedding(pieces)
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).reshape(batch_size, -1, 4, self.d_model)
        pieces_visible = pieces_with_rotations[:, :7]
        pieces_not_visible = pieces_with_rotations[:, 7:]

        garbage_processed = self.garbage_embedding(garbage.unsqueeze(-1))
        garbage_processed = self.garbage_embedding2(
            torch.max(1)
        )

        board_processed = self.CNN(board)

        state = board_processed+garbage_processed
        if self.retain is not None:
            state = state+self.retain

        def attn(state, pieces):
            num_heads = 8
            d_head = self.d_model // num_heads
            # Linear transformations
            q = self.q_linear(state).view(batch_size, 18,
                                          num_heads, d_head).transpose(1, 2)
            k = self.k_linear(pieces_visible[:, i]).view(
                batch_size, -1, num_heads, d_head).transpose(1, 2)
            v = self.v_linear(pieces_visible[:, i]).view(
                batch_size, -1, num_heads, d_head).transpose(1, 2)

            bias = self.attn_bias(state)
            # Scaled dot-product attention
            attn_weights = torch.matmul(
                q, k.transpose(-2, -1)) / (d_head ** 0.5) + bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(attn_weights, v)

            # Combine heads
            output = output.view(batch_size, 18 * self.d_model)

        for i in range(7):
            new_state = attn(state, pieces_visible[:, i:i+1])
            state = self.layer_norm(state+new_state)

            new_state = self.remain_linear(state)
            state = self.layer_norm(state+new_state)
        output = attn(state, pieces_not_visible)
        state = self.layer_norm(state+new_state)

        new_state = self.queue_linear(state)
        state = self.layer_norm(state+new_state)
        self.retain = output
        policy = self.policy(output)
        value = self.value(output)
        return policy, value


class Bias_Layer(nn.Module):
    def __init__(self, ff_size, d_model):
        super(Bias_Layer, self).__init__()
        self.relu6 = nn.ReLU6()
        self.linear1 = nn.Linear(d_model, 32, bias=False)
        self.linear2 = nn.Linear(32, 256)
        self.layernorm1 = nn.LayerNorm(256)
        self.linear3 = nn.Linear(256, 256*8)
        self.layernorm2 = nn.LayerNorm(256)
        self.linear4 = nn.Linear(256, ff_size)

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
    def __init__(self, d_model):
        super(SimpleCNN, self).__init__()

        # First layer: Depthwise separable convolution
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, groups=3)
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
        x = F.pad(x, (0, 0, 0, 1))
        x = self.pool(x)

        return x
