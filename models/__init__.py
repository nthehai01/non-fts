import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01) 


class CellEncoder(nn.Module):
    def __init__(self, model_path):
        super(CellEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        

    def forward(self, input_ids, attention_masks):
        batch_size, max_cell, max_len = input_ids.shape

        input_ids = input_ids.view(-1, max_len)
        attention_masks = attention_masks.view(-1, max_len)

        tokens = self.bert(input_ids, attention_masks)['last_hidden_state']
        emb_dim = tokens.shape[-1]  # bert output embedding dim
        tokens = tokens.view(batch_size, max_cell, max_len, emb_dim)

        return tokens


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.W = Linear(dim, dim, bias=False)
        self.v = Linear(dim, 1, bias=False)
        

    def forward(self, keys, masks):
        weights = self.v(torch.tanh(self.W(keys)))
        weights.masked_fill_(masks.unsqueeze(-1).bool(), -6.5e4)
        weights = F.softmax(weights, dim=2)
        return torch.sum(weights * keys, dim=2)


class PositionalEncoder(nn.Module):
    def __init__(self):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=0.1)


    def _create_angle_rates(self, dim):
        angles = torch.arange(dim)
        angles[1::2] = angles[0::2]
        angles = 1 / (10000 ** (angles / dim))
        angles = torch.unsqueeze(angles, axis=0)
        return angles


    def _generate_positional_encoding(self, pos, d_model):
        angles = self._create_angle_rates(d_model).type(torch.float32)
        pos = torch.unsqueeze(torch.arange(pos), axis=1).type(torch.float32)
        pos_angles = torch.matmul(pos, angles)
        pos_angles[:, 0::2] = torch.sin(pos_angles[:, 0::2])
        pos_angles[:, 1::2] = torch.cos(pos_angles[:, 1::2])
        pos_angles = torch.unsqueeze(pos_angles, axis=0)

        return pos_angles


    def forward(self, x):
        _, max_cell, emb_dim = x.shape

        pos_encoding = self._generate_positional_encoding(max_cell, emb_dim)

        x += pos_encoding[:, :max_cell, :].to(x.device)
        x = self.dropout(x)
        return x
