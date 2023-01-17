import torch
import torch.nn as nn
from transformers import AutoConfig
import os

from models import Linear, CellEncoder


NON_MASKED_INDEX = int(os.environ['NON_MASKED_INDEX'])


class NotebookMLM(nn.Module):
    def __init__(self, 
                 code_pretrained, 
                 md_pretrained, 
                 n_heads, 
                 n_layers):
        super(NotebookMLM, self).__init__()
        code_pretrained_config = AutoConfig.from_pretrained(code_pretrained)
        code_emb_dim = code_pretrained_config.hidden_size
        md_pretrained_config = AutoConfig.from_pretrained(md_pretrained)
        md_emb_dim = md_pretrained_config.hidden_size

        self.code_encoder = CellEncoder(code_pretrained)
        self.md_encoder = CellEncoder(md_pretrained)
        self.code_cell_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=code_emb_dim, nhead=n_heads, batch_first=True), 
            num_layers=n_layers
        )
        self.md_cell_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=md_emb_dim, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.fc = Linear(md_pretrained_config.hidden_size, md_pretrained_config.vocab_size, bias=False)


    def forward(self, 
                code_input_ids, code_attention_masks, 
                md_input_ids, md_attention_masks,
                code_cell_padding_masks, md_cell_padding_masks):
        # cell encoder
        code_embedding = self.code_encoder(code_input_ids, code_attention_masks)  # [..., max_code_cell, max_len, emb_dim]
        md_embedding = self.md_encoder(md_input_ids, md_attention_masks)  # [..., max_md_cell, max_len, emb_dim]

        # reshape
        batch_size = code_embedding.shape[0]
        max_len = code_embedding.shape[-2]
        emd_dim = code_embedding.shape[-1]
        
        code_embedding = code_embedding.view(batch_size, -1, emd_dim)  # [..., max_code_cell*max_len, emb_dim]
        md_embedding = md_embedding.view(batch_size, -1, emd_dim)  # [..., max_md_cell*max_len, emb_dim]
        
        code_cell_padding_masks = torch.unsqueeze(code_cell_padding_masks, dim=-1)  # [..., max_code_cell, 1]
        code_cell_padding_masks  = code_cell_padding_masks.repeat(1, 1, max_len)  # [..., max_code_cell, max_len]
        code_cell_padding_masks = code_cell_padding_masks.view(batch_size, -1)  # [..., max_code_cell*max_len]

        md_cell_padding_masks = torch.unsqueeze(md_cell_padding_masks, dim=-1)  # [..., max_md_cell, 1]
        md_cell_padding_masks  = md_cell_padding_masks.repeat(1, 1, max_len)  # [..., max_md_cell, max_len]
        md_cell_padding_masks = md_cell_padding_masks.view(batch_size, -1)  # [..., max_md_cell*max_len]

        # code cell encoder
        code_cell_embedding = self.code_cell_encoder(
            src=code_embedding,
            src_key_padding_mask=code_cell_padding_masks
        )  # [..., max_code_cell*max_len, emb_dim]

        # md cell decoder
        md_cell_embedding = self.md_cell_decoder(
            tgt=md_embedding,
            memory=code_cell_embedding,
            tgt_key_padding_mask=md_cell_padding_masks,
            memory_key_padding_mask=code_cell_padding_masks
        )  # [..., max_md_cell*max_len, emb_dim]

        prediction_scores = self.fc(md_cell_embedding)  # [..., max_md_cell*max_len, vocab_size]
        
        return prediction_scores
