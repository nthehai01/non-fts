import torch
import torch.nn as nn
from transformers import AutoConfig

from models import Linear, CellEncoder, Attention, PositionalEncoder


class PointwiseHead(nn.Module):
    def __init__(self, dim):
        super(PointwiseHead, self).__init__()
        self.fc0 = Linear(dim, 256)
        self.fc1 = Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.top = Linear(128, 1)  
        
        
    def forward(self, x, md_pct):
        x = x[:, 1:-1]

        x = self.fc0(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.fc1(x)
        x = self.act(x)

        x = self.top(x)

        return x.squeeze(-1)


class NotebookOrdering(nn.Module):
    def __init__(self, code_pretrained, md_pretrained, n_heads, n_layers):
        super(NotebookOrdering, self).__init__()
        code_pretrained_config = AutoConfig.from_pretrained(code_pretrained)
        code_emb_dim = code_pretrained_config.hidden_size
        md_pretrained_config = AutoConfig.from_pretrained(md_pretrained)
        md_emb_dim = md_pretrained_config.hidden_size

        self.code_encoder = CellEncoder(code_pretrained)
        self.md_encoder = CellEncoder(md_pretrained)
        self.code_pooling = Attention(code_emb_dim)
        self.md_pooling = Attention(md_emb_dim)
        self.code_positional_encoder = PositionalEncoder()
        self.code_cell_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=code_emb_dim, nhead=n_heads, batch_first=True), 
            num_layers=n_layers
        )
        self.md_cell_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=md_emb_dim, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.pointwise_head = PointwiseHead(md_emb_dim)


    def forward(self, 
                code_input_ids, code_attention_masks, 
                md_input_ids, md_attention_masks,
                code_cell_padding_masks, md_cell_padding_masks,
                md_pct):
        # cell encoder
        code_embedding = self.code_encoder(code_input_ids, code_attention_masks)  # [..., max_code_cell+2, max_len, emb_dim]
        md_embedding = self.md_encoder(md_input_ids, md_attention_masks)  # [..., max_md_cell+2, max_len, emb_dim]

        # cell pooling
        code_embedding = self.code_pooling(code_embedding, code_attention_masks)  # [..., max_code_cell+2, emb_dim]
        md_embedding = self.md_pooling(md_embedding, md_attention_masks)  # [..., max_md_cell+2, emb_dim]

        # add positional encoder
        code_embedding = self.code_positional_encoder(code_embedding)  # [..., max_code_cell+2, emb_dim]

        # code cell encoder
        code_cell_embedding = self.code_cell_encoder(
            src=code_embedding,
            src_key_padding_mask=code_cell_padding_masks
        )  # [..., max_code_cell+2, emb_dim]

        # md cell decoder
        md_cell_embedding = self.md_cell_decoder(
            tgt=md_embedding,
            memory=code_cell_embedding,
            tgt_key_padding_mask=md_cell_padding_masks,
            memory_key_padding_mask=code_cell_padding_masks
        )  # [..., max_md_cell+2, emb_dim]

        # pointwise head
        x = self.pointwise_head(md_cell_embedding, md_pct)  # [..., max_md_cell]

        return x
