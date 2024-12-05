import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import args
seq_len = args.seq_len
ff_size = args.ff_size
d_model = args.d_model


class Model3(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(Model3, self).__init__()
        self.d_model = d_model
        self.ff_size = ff_size
        self.relu6 = nn.ReLU6()
        self.ppos = PositionalEncoding(d_model, max_len=7)
        self.loc_embedding = nn.Linear(4, d_model)
        self.loc_embedding2 = nn.Linear(d_model, d_model)
        self.piece_embedding = nn.Embedding(8, d_model)
        self.rotation_embedding = nn.Linear(d_model, d_model*4)

        self.CNN = SimpleCNN(d_model)
        self.piece_board_attn = AttnWithBias(seq_len, 56, d_model, 4)
        # Separate layers for each decoder
        macro_layer = nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=ff_size,
            nhead=4, batch_first=True)
        board_layer = nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=ff_size,
            nhead=4, batch_first=True)
        movement_layer = nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=ff_size,
            nhead=4, batch_first=True)

        # Separate decoders with their own layers
        self.macro_decoder = nn.TransformerDecoder(
            macro_layer, num_layers=2)
        self.board_decoder = nn.TransformerDecoder(
            board_layer, num_layers=2)
        self.movement_decoder = nn.TransformerDecoder(
            movement_layer, num_layers=2)

        self.p_encode = PositionalEncoding2D(d_model, 20, 10)

        self.final_linear = nn.Linear(d_model, ff_size)
        self.final_linear2 = nn.Linear(ff_size, ff_size)
        self.final_norm = nn.LayerNorm(ff_size)
        self.policy = MLP((ff_size, ff_size, 10))
        self.value = nn.Linear(ff_size, ff_size)

    def forward(self, x, state_in):
        xyrot, queue, remains, board, _ = x
        batch_size = queue.size(0)

        default_pos = torch.full(
            (batch_size, 13, 4), -1, device=torch.device("cuda"))
        pos = torch.cat((xyrot.unsqueeze(1), default_pos), 1)
        loc = self.loc_embedding(pos)
        loc = self.relu6(loc)
        loc_processed = self.loc_embedding2(loc)

        pieces = torch.cat((queue, remains), 1)
        pieces_processed = self.piece_embedding(pieces)
        pieces_processed = pieces_processed + loc_processed
        pieces_processed = self.ppos(pieces_processed)
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).view(batch_size, -1, d_model)
        pieces_with_rotations = self.relu6(pieces_with_rotations)

        board_processed = self.CNN(board.float().unsqueeze(1))  # bchw
        board_with_position = self.p_encode(
            board_processed).flatten(-2).swapaxes(-1, -2).contiguous()
        assert board_with_position.shape[1:] == (seq_len, d_model)
        pb = self.piece_board_attn(
            board_with_position,
            pieces_with_rotations)

        state = self.macro_decoder(pb, state_in)

        new_state = self.board_decoder(
            state, pieces_with_rotations)

        state = new_state  # state+new_state
        x = self.movement_decoder(pieces_processed[:,:1], state
                                  )
        x = x.view(batch_size, d_model)
        # x = x+loc_processed

        x = self.final_linear(x)
        x = self.relu6(x)
        x = self.final_linear2(x)+x
        x = self.final_norm(x)

        logits = self.policy(x)
        value_latent = self.relu6(self.value(x))
        return logits, value_latent


class NoMacroModel(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(NoMacroModel, self).__init__()

        self.value = MLP((ff_size, ff_size,  1))
        self.model = Model3()

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0,
                                std=np.sqrt(1. / module.weight.size(1)))
        nn.init.orthogonal_(self.model.policy.layers[-1].weight, 0.1)
        nn.init.orthogonal_(self.value.layers[-1].weight, 1)

    def forward(self, x, y, memory=None, mask=None):
        policy, value = self.model(x)

        final_value = self.value(value).squeeze()
        return policy, final_value, 0


class MacroModel(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(MacroModel, self).__init__()
        self.relu6 = nn.ReLU6()
        self.cnn = SimpleCNN(ff_size)
        self.queue_linear = nn.Linear(14, ff_size)
        self.piece_embedding = nn.Embedding(8, ff_size)
        self.ppos = PositionalEncoding(ff_size, max_len=7)
        self.garbage_embedding1 = nn.Conv1d(
            in_channels=1, out_channels=d_model, kernel_size=3, padding=1, stride=3)
        self.garbage_embedding2 = nn.Conv1d(
            in_channels=d_model, out_channels=ff_size, kernel_size=2, padding=1, stride=2)
        self.garbage_embedding3 = nn.Conv1d(
            in_channels=ff_size, out_channels=ff_size, kernel_size=2, stride=2)

        self.own_attn = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.opp_attn = nn.MultiheadAttention(ff_size, 8, batch_first=True)

        self.mlp = MLP((ff_size*3, ff_size, 4*d_model))
        self.latent = nn.Linear(d_model, d_model)
        self.initial_tokens = nn.Parameter(torch.randn(6, d_model))
        memory_layer = nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=ff_size,
            nhead=4, batch_first=True)
        self.memory_decoder = nn.TransformerDecoder(
            memory_layer, num_layers=2)
        state_layer = nn.TransformerDecoderLayer(
            d_model=d_model, dim_feedforward=ff_size,
            nhead=4, batch_first=True)
        self.state_decoder = nn.TransformerDecoder(
            state_layer, num_layers=2)
        self.mem_gate = MLP((d_model, ff_size, d_model))
        self.macro_value = nn.Linear(d_model, ff_size)
        self.value = MLP((ff_size*2, ff_size,  1))
        self.model = Model3()

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0,
                                std=np.sqrt(1. / module.weight.size(1)))
        nn.init.orthogonal_(self.model.policy.layers[-1].weight, 0.01)
        nn.init.orthogonal_(self.value.layers[-1].weight, 1)

    def forward(self, x, y, memory, mask, out_mask=None):
        _, queue, remains, board, garbage = x
        _, opp_q, opp_remain, opp_board, _ = y
        batch_size = queue.size(0)

        pieces = torch.cat((queue, remains), 1)
        pieces_processed = self.piece_embedding(pieces)
        pieces_processed = self.ppos(pieces_processed)

        opp_pieces = torch.cat((opp_q, opp_remain), 1)
        opp_p = self.piece_embedding(opp_pieces)
        opp_p = self.ppos(opp_p)

        garbage_processed = self.garbage_embedding1(
            garbage.unsqueeze(1).float())
        garbage_processed = self.garbage_embedding2(
            garbage_processed
        )
        garbage_processed = self.garbage_embedding3(
            garbage_processed
        ).view(batch_size, ff_size)
        own_board = self.cnn(board.unsqueeze(1)).view(
            batch_size, ff_size, -1).swapaxes(1, 2)
        own_state, _ = self.own_attn(
            own_board, pieces_processed, pieces_processed)
        opp_board = self.cnn(opp_board.unsqueeze(1)).view(
            batch_size, ff_size, -1).swapaxes(1, 2)
        opp_state, _ = self.opp_attn(opp_board, opp_p, opp_p)

        state = torch.cat(
            (own_state.mean(1), opp_state.mean(1), garbage_processed), 1)
        state = self.mlp(state).view(batch_size, -1, d_model)
        memory[mask] = self.initial_tokens
        new_memory = self.memory_decoder(memory, state)
        new_memory = self.mem_gate(new_memory+memory)
        if out_mask is not None:
            new_memory[out_mask] = self.initial_tokens
            return new_memory
        new_state = self.state_decoder(state, memory)
        state = state+new_state
        value2 = self.macro_value(state.mean(1))
        value2 = self.relu6(value2)

        latent = self.latent(state)
        policy, value = self.model(x, latent)

        value = torch.cat((value, value2), -1)
        final_value = self.value(value).squeeze()
        return policy, final_value, new_memory


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU6()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class Model(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(Model, self).__init__()
        self.d_model = d_model
        self.ff_size = ff_size
        self.loc_embedding = nn.Linear(4, 16)
        self.loc_embedding2 = nn.Linear(16, ff_size)
        self.piece_embedding = nn.Embedding(8, ff_size)
        self.rotation_embedding = nn.Linear(ff_size, ff_size*4)

        self.garbage_embedding = nn.Conv1d(
            in_channels=1, out_channels=ff_size, kernel_size=3, padding=1)
        self.garbage_embedding2 = nn.Linear(10, seq_len)
        self.q_linear = nn.Linear(ff_size, d_model)
        self.k_linear = nn.Linear(ff_size, d_model)
        self.v_linear = nn.Linear(ff_size, d_model)
        self.qkv_linear = nn.Linear(d_model, ff_size)
        self.attn_bias = Bias_Layer(ff_size, seq_len, 1)
        self.attn_last = Bias_Layer(ff_size, seq_len, 7)

        self.queue_linear = nn.Linear(ff_size, ff_size)
        self.attn_norm = nn.LayerNorm(ff_size)
        self.ff_norm = nn.LayerNorm(ff_size)

        self.CNN = SimpleCNN(ff_size)
        self.p_encode = PositionalEncoding2D(ff_size, 20, 10)

        self.attn = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.policy = nn.Linear(ff_size, 10)
        self.value = nn.Linear(ff_size, ff_size)

    def forward(self, x, state_in=0):
        xyrot, queue, remains, board, garbage = x
        batch_size = queue.size(0)

        loc = self.loc_embedding(xyrot)
        loc_processed = self.loc_embedding2(loc)

        pieces = torch.cat((queue, remains), 1)
        pieces_processed = self.piece_embedding(pieces)
        current = pieces_processed[:, 0] + loc_processed
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).view(batch_size, -1, 4, self.ff_size)
        pieces_visible = pieces_with_rotations[:, :7]
        pieces_not_visible = pieces_with_rotations[:, 7:]

        board_processed = self.CNN(board.float().unsqueeze(1))  # bchw
        board_with_position = self.p_encode(
            board_processed).flatten(-2).swapaxes(-1, -2).contiguous()
        assert board_with_position.shape[1:] == (seq_len, ff_size)

        state = board_with_position+state_in

        def attn(state, pieces, layer=None):
            num_heads = 8
            d_head = self.d_model // num_heads
            # Linear transformations
            q = self.q_linear(state).view(batch_size, seq_len,
                                          num_heads, d_head)
            k = self.k_linear(pieces).view(
                batch_size, -1, num_heads, d_head)
            v = self.v_linear(pieces).view(
                batch_size, -1, num_heads, d_head)
            if layer:
                bias = layer(state).view(
                    batch_size, num_heads, seq_len, k.size(1)).transpose(1, 2)
            else:
                bias = 0

            # Scaled dot-product attention
            attn_weights = torch.einsum(
                'bqnd,bknd->bqnk', q, k) / (d_head ** 0.5)
            attn_weights = F.softmax(attn_weights + bias, dim=-1)
            output = torch.einsum('bqnk,bvnd->bqnd', attn_weights, v)

            # Combine heads
            output = output.view(batch_size, seq_len, self.d_model)
            return self.qkv_linear(output)

        for i in range(7):
            new_state = attn(state, pieces_visible[:, i:i+1], self.attn_bias)
            state = self.attn_norm(state+new_state)

            new_state = self.queue_linear(state)
            state = self.ff_norm(state+new_state)
        new_state = attn(state, pieces_not_visible, self.attn_last)
        state = self.attn_norm(state+new_state)

        new_state = self.queue_linear(state)
        state = self.ff_norm(state+new_state)

        x, _ = self.attn(current.unsqueeze(1), state, state)
        x = x.view(batch_size, ff_size)

        logits = self.policy(x)
        value_latent = self.value(x)
        return logits, value_latent


class SimpleModel(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(SimpleModel, self).__init__()
        self.d_model = d_model
        self.ff_size = ff_size
        self.loc_embedding = nn.Linear(4, ff_size//4)
        self.loc_embedding2 = nn.Linear(ff_size//4, ff_size)
        self.piece_embedding = nn.Embedding(8, ff_size)
        self.rotation_embedding = nn.Linear(ff_size, ff_size*4)

        self.q_linear = nn.Linear(ff_size, d_model)
        self.k_linear = nn.Linear(ff_size, d_model)
        self.v_linear = nn.Linear(ff_size, d_model)
        self.qkv_linear = nn.Linear(d_model, ff_size)

        self.queue_linear = nn.Linear(ff_size, ff_size)
        self.attn_norm = nn.LayerNorm(ff_size)
        self.ff_norm = nn.LayerNorm(ff_size)

        self.CNN = SimpleCNN(ff_size)
        self.p_encode = PositionalEncoding2D(ff_size, 20, 10)

        self.attn0 = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.attn = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.policy = nn.Linear(ff_size, 10)
        self.value = nn.Linear(ff_size, ff_size)

    def forward(self, x, state_in=0, memory=0):
        xyrot, queue, _, board, _ = x
        batch_size = queue.size(0)

        loc = self.loc_embedding(xyrot)
        loc_processed = self.loc_embedding2(loc)

        pieces = queue[:, 0]
        pieces_processed = self.piece_embedding(pieces)
        current = pieces_processed + loc_processed
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).view(batch_size, -1, ff_size)

        board_processed = self.CNN(board.float().unsqueeze(1))  # bchw
        board_with_position = self.p_encode(
            board_processed).flatten(-2).swapaxes(-1, -2).contiguous()
        assert board_with_position.shape[1:] == (seq_len, ff_size)

        state_in = state_in.unsqueeze(1)
        state, _ = self.attn0(board_with_position, state_in, state_in)

        def attn(state, pieces, layer=None):
            num_heads = 8
            d_head = self.d_model // num_heads
            # Linear transformations
            q = self.q_linear(state).view(batch_size, seq_len,
                                          num_heads, d_head)
            k = self.k_linear(pieces).view(
                batch_size, -1, num_heads, d_head)
            v = self.v_linear(pieces).view(
                batch_size, -1, num_heads, d_head)
            if layer:
                bias = layer(state, pieces).view(
                    batch_size, num_heads, seq_len, k.size(1)).transpose(1, 2)
            else:
                bias = 0

            # Scaled dot-product attention
            attn_weights = torch.einsum(
                'bqnd,bknd->bqnk', q, k) / (d_head ** 0.5)
            attn_weights = F.softmax(attn_weights + bias, dim=-1)
            output = torch.einsum('bqnk,bvnd->bqnd', attn_weights, v)

            # Combine heads
            output = output.view(batch_size, seq_len, self.d_model)
            return self.qkv_linear(output)
        current = current.unsqueeze(1)
        new_state = attn(state, pieces_with_rotations)
        state = self.attn_norm(state+new_state)

        new_state = self.queue_linear(state)
        state = self.ff_norm(state+new_state)

        x, _ = self.attn(current, state, state)
        x = x.view(batch_size, ff_size)+memory

        logits = self.policy(x)
        value_latent = self.value(x)
        return logits, value_latent


class Model2(nn.Module):
    def __init__(self, ff_size=ff_size, d_model=d_model):
        super(Model2, self).__init__()
        self.d_model = d_model
        self.ff_size = ff_size
        self.relu6 = nn.ReLU6()
        self.ppos = PositionalEncoding(ff_size, max_len=7)
        self.loc_embedding = nn.Linear(4, ff_size//4)
        self.loc_embedding2 = nn.Linear(ff_size//4, ff_size)
        self.piece_embedding = nn.Embedding(8, ff_size)
        self.rotation_embedding = nn.Linear(ff_size, ff_size*4)

        self.attn0 = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.self_attn = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.attn1 = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(ff_size, 8, batch_first=True)
        self.qkv_linear = nn.Linear(ff_size, ff_size)

        self.combine_linear = nn.Linear(ff_size, ff_size)
        self.combine_norm = nn.LayerNorm(ff_size)
        self.self_linear1 = nn.Linear(ff_size, ff_size)
        self.self_norm1 = nn.LayerNorm(ff_size)
        self.self_linear2 = nn.Linear(ff_size, ff_size)
        self.self_norm2 = nn.LayerNorm(ff_size)
        self.self_linear21 = nn.Linear(ff_size, ff_size)
        self.self_norm21 = nn.LayerNorm(ff_size)
        self.self_linear22 = nn.Linear(ff_size, ff_size)
        self.self_norm22 = nn.LayerNorm(ff_size)
        self.queue_linear = nn.Linear(ff_size, ff_size)
        self.attn_norm = nn.LayerNorm(ff_size)
        self.ff_norm = nn.LayerNorm(ff_size)

        self.CNN = SimpleCNN(ff_size)
        self.p_encode = PositionalEncoding2D(ff_size, 20, 10)

        self.final_linear = nn.Linear(ff_size, ff_size)
        self.policy = MLP((ff_size, ff_size, 10))
        self.value = nn.Linear(ff_size, ff_size)

    def forward(self, x, state_in=0, memory=0):
        xyrot, queue, remains, board, _ = x
        batch_size = queue.size(0)

        loc = self.loc_embedding(xyrot)
        loc = self.relu6(loc)
        loc_processed = self.loc_embedding2(loc)

        pieces = torch.cat((queue, remains), 1)
        pieces_processed = self.piece_embedding(pieces)
        current = pieces_processed[:, 0] + loc_processed
        pieces_processed = self.ppos(pieces_processed)
        pieces_with_rotations = self.rotation_embedding(
            pieces_processed).view(batch_size, -1, ff_size)
        pieces_with_rotations = self.relu6(pieces_with_rotations)

        board_processed = self.CNN(board.float().unsqueeze(1))  # bchw
        board_with_position = self.p_encode(
            board_processed).flatten(-2).swapaxes(-1, -2).contiguous()
        assert board_with_position.shape[1:] == (seq_len, ff_size)

        state_in = state_in.unsqueeze(1)
        state, _ = self.attn0(board_with_position, state_in, state_in)
        state = self.combine_linear(state)
        state = self.combine_norm(state)
        state = self.relu6(state)

        new_state, _ = self.self_attn(state, state, state)
        state = self.self_linear1(state)+state
        state = self.self_norm1(state)
        state = self.self_linear2(state)+state
        state = self.self_norm2(state)
        state = self.relu6(state)

        new_state, _ = self.attn1(
            state, pieces_with_rotations, pieces_with_rotations)
        new_state = self.qkv_linear(new_state)
        state = self.attn_norm(state+new_state)
        state = self.relu6(state)

        new_state = self.queue_linear(state)
        state = self.ff_norm(state+new_state)
        state = self.relu6(state)

        new_state, _ = self.self_attn2(state, state, state)
        state = self.self_linear21(state)+state
        state = self.self_norm21(state)
        state = self.self_linear22(state)+state
        state = self.self_norm22(state)
        state = self.relu6(state)

        current = current.unsqueeze(1)
        x, _ = self.attn2(current, state, state)
        x = x.view(batch_size, ff_size)+memory
        x = self.final_linear(x)
        x = self.relu6(x)

        logits = self.policy(x)
        value_latent = self.relu6(self.value(x))
        return logits, value_latent


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 7):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len*2, 1, d_model)
        pe[:max_len, 0, 0::2] = torch.sin(position * div_term)
        pe[:max_len, 0, 1::2] = torch.cos(position * div_term)
        pe[max_len:, 0, 0::2] = torch.sin(max_len * div_term)
        pe[max_len:, 0, 1::2] = torch.cos(max_len * div_term)
        pe = pe.swapaxes(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(d_model, height, width)
        for y in range(height):
            for x in range(width):
                for i in range(0, d_model, 2):
                    pe[i, y, x] = np.sin(y / (10000 ** ((2 * i) / d_model)))
                    pe[i+1, y, x] = np.cos(x / (10000 ** ((2 * i) / d_model)))

        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class AttnWithBias(nn.Module):
    def __init__(self, len1, len2, d_model,  num_heads=8):
        super(AttnWithBias, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = self.d_model // num_heads
        self.len1 = len1
        self.len2 = len2
        self.biaslayer = Bias_Layer(len1, len1, len2, num_heads)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, ff_size)
        self.linear2 = nn.Linear(ff_size, d_model)
        self.activation = nn.ReLU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, state, pieces, layer=None):
        batch_size = state.size(0)
        # Linear transformations
        q = self.q_linear(state).view(batch_size, self.len1,
                                      self.num_heads, self.d_head)
        k = self.k_linear(pieces).view(
            batch_size, self.len2, self.num_heads, self.d_head)
        v = self.v_linear(pieces).view(
            batch_size, self.len2, self.num_heads, self.d_head)
        bias = self.biaslayer(state, pieces).view(
            batch_size, self.num_heads, self.len1, self.len2).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.einsum(
            'bqnd,bknd->bqnk', q, k) / (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_weights + bias, dim=-1)
        output = torch.einsum('bqnk,bvnd->bqnd', attn_weights, v)

        # Combine heads
        output = output.view(batch_size, self.len1, self.d_model)
        state = self.norm1(state+output)
        new_state = self.activation(self.linear1(state))
        state = self.norm2(state+self.linear2(new_state))
        return state


class Bias_Layer(nn.Module):
    def __init__(self, in_size, seq_len, out_len, nhead):
        super(Bias_Layer, self).__init__()
        self.hidden_size = 128
        self.nhead = nhead
        self.relu6 = nn.ReLU6()
        self.linear1 = nn.Linear(256, 32, bias=False)
        self.linear2 = nn.Linear(32*d_model, self.hidden_size)
        self.layernorm1 = nn.LayerNorm(self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size*nhead)
        self.layernorm2 = nn.LayerNorm(self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, seq_len*out_len)
        self.seq_len = seq_len

    def forward(self, x, y):
        x = torch.cat((x, y), 1).swapaxes(-1, -2)
        x = self.linear1(x).swapaxes(-1, -2).reshape(x.size(0), 32*d_model)
        x = self.relu6(x)

        x = self.linear2(x)
        x = self.layernorm1(x)
        x = self.relu6(x)

        x = self.linear3(x).view(-1, self.nhead, self.hidden_size)
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
            in_channels=1, out_channels=out_size//2, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_size//2)
        self.relu6 = nn.ReLU6()

        # Second layer: Depthwise separable convolution
        self.depthwise_conv2 = nn.Conv2d(in_channels=out_size//2, out_channels=out_size//2,
                                         kernel_size=3, padding=1, groups=out_size//2)
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=out_size//2, out_channels=out_size, kernel_size=1)

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

        return x
