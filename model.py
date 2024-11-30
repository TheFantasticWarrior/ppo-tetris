import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import args
seq_len = args.seq_len
ff_size = args.ff_size
d_model = args.d_model


class Model(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(Model, self).__init__()
        self.d_model = d_model
        self.loc_embedding = nn.Linear(3, 128)
        self.loc_embedding2 = nn.Linear(128, d_model)
        self.piece_embedding = nn.Embedding(8, d_model)
        self.rotation_embedding = nn.Linear(d_model, d_model*4)

        self.garbage_embedding = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.garbage_embedding2 = nn.Linear(10, seq_len)

        self.q_linear = nn.ModuleList(
            [nn.Linear(ff_size, d_model) for _ in range(8)])
        self.k_linear = nn.ModuleList(
            [nn.Linear(d_model, d_model//4) for _ in range(8)])
        self.v_linear = nn.ModuleList(
            [nn.Linear(d_model, d_model//4) for _ in range(8)])
        self.qkv_linear = nn.ModuleList(
            [nn.Linear(d_model, ff_size) for _ in range(8)])
        self.attn_bias = nn.ModuleList(
            [Bias_Layer(seq_len*ff_size, seq_len) for _ in range(7)] +
            [Bias_Layer(seq_len*ff_size, seq_len*7)])

        self.queue_linear = nn.ModuleList(
            [nn.Linear(ff_size, ff_size) for _ in range(7)])
        self.remain_linear = nn.Linear(ff_size, ff_size)
        self.attn_norm = nn.ModuleList(
            [nn.LayerNorm(ff_size) for _ in range(8)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(ff_size) for _ in range(8)])

        self.CNN = SimpleCNN(ff_size)

        self.last_linear = nn.Linear(ff_size, d_model)

        self.p_encode = PositionalEncoding(ff_size)
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d)\
                    or isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, np.sqrt(2))

    def forward(self, x):
        xyrot, queue, remains, board, garbage = x
        batch_size = queue.size(0)

        loc = self.loc_embedding(xyrot)
        loc_processed = self.loc_embedding2(loc)

        pieces = torch.cat((queue, remains), 1)
        pieces_processed = self.piece_embedding(pieces)
        pieces_processed[:, 0] += loc_processed
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).view(batch_size, -1, 4, self.d_model)
        pieces_visible = pieces_with_rotations[:, :7]
        pieces_not_visible = pieces_with_rotations[:, 7:]

        garbage_processed = self.garbage_embedding(
            garbage.unsqueeze(1).float())
        garbage_processed = self.garbage_embedding2(
            garbage_processed
        ).swapaxes(-1, -2)

        board_processed = self.CNN(board.float().unsqueeze(1))
        assert board_processed.shape[1:] == (seq_len, ff_size)
        board_with_position = self.p_encode(board_processed)

        state = board_with_position+garbage_processed

        def attn(state, pieces, i):
            num_heads = 8
            d_head = self.d_model // num_heads
            # Linear transformations
            q = self.q_linear[i](state).view(batch_size, seq_len,
                                             num_heads, d_head)
            k = self.k_linear[i](pieces).view(
                batch_size, -1, num_heads, d_head)
            v = self.v_linear[i](pieces).view(
                batch_size, -1, num_heads, d_head)
            bias = self.attn_bias[i](state.view(batch_size, -1)).view(
                batch_size, num_heads, seq_len, k.size(1)).transpose(1, 2)

            # Scaled dot-product attention
            attn_weights = torch.einsum(
                'bqnd,bknd->bqnk', q, k) / (d_head ** 0.5)
            attn_weights = F.softmax(attn_weights + bias, dim=-1)
            output = torch.einsum('bqnk,bvnd->bqnd', attn_weights, v)

            # Combine heads
            output = output.view(batch_size, seq_len, self.d_model)
            return self.qkv_linear[i](output)

        for i in range(7):
            new_state = attn(state, pieces_visible[:, i:i+1], i)
            state = self.attn_norm[i](state+new_state)

            new_state = self.queue_linear[i](state)
            state = self.ff_norm[i](state+new_state)
        new_state = attn(state, pieces_not_visible, 7)
        state = self.attn_norm[-1](state+new_state)

        new_state = self.remain_linear(state)
        state = self.ff_norm[-1](state+new_state)
        state = self.last_linear(state)
        return state


class FinalModel(nn.Module):
    def __init__(self, ff_size=128, d_model=d_model):
        super(FinalModel, self).__init__()
        self.d_model = d_model
        self.attn1 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.t_encoder = nn.Transformer(
            d_model=d_model, nhead=8, dim_feedforward=ff_size,
            batch_first=True, num_encoder_layers=3, num_decoder_layers=0
        ).encoder
        self.linear = nn.Linear(d_model*200, ff_size)
        self.policy = nn.Linear(ff_size, 10)
        self.value = nn.Linear(ff_size, 1)
        self.memory = nn.Linear(seq_len, 8)

    def forward(self, x, y, retain):
        batch_size = x.size(0)
        combined, _ = self.attn1(x, y, y)
        x = combined+x
        x_mem_combined, _ = self.attn2(x, retain, retain)
        x = x_mem_combined+x
        x = self.t_encoder(x)
        memory = self.memory(x.swapaxes(1, 2)).swapaxes(1, 2)
        x = self.linear(x.view(batch_size, -1))
        logits = self.policy(x)
        value = self.value(x).squeeze()
        return logits, value, memory


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.swapaxes(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Bias_Layer(nn.Module):
    def __init__(self, in_size, out_size):
        super(Bias_Layer, self).__init__()
        self.relu6 = nn.ReLU6()
        self.linear1 = nn.Linear(in_size, 32, bias=False)
        self.linear2 = nn.Linear(32, 256)
        self.layernorm1 = nn.LayerNorm(256)
        self.linear3 = nn.Linear(256, 256*8)
        self.layernorm2 = nn.LayerNorm(256)
        self.linear4 = nn.Linear(256, out_size)

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
    def __init__(self, out_size):
        super(SimpleCNN, self).__init__()

        # First layer: Depthwise separable convolution
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3)
        self.pointwise_conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU6()

        # Second layer: Depthwise separable convolution
        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16,
                                         kernel_size=3, padding=1, groups=16)
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=16, out_channels=out_size, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)

        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn2(x)
        x = self.relu6(x)

        x = x.flatten(-2).swapaxes(-1, -2).contiguous()
        return x
