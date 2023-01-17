import torch

from datasets.ordering_dataset import OrderingDataset
from utils import encode_texts, extract_nb_info, mask_tokens


class MLMDataset(OrderingDataset):
    def __init__(self, 
                 code_pretrained, 
                 md_pretrained, 
                 max_len, 
                 ellipses_token_id, 
                 df_id,
                 nb_meta_data, 
                 df_code_cell,
                 df_md_cell,
                 max_n_code_cells,
                 max_n_md_cells,
                 mlm_probability=0.15,
                 is_train=False,):
        super().__init__(code_pretrained, 
                         md_pretrained, 
                         max_len, 
                         ellipses_token_id, 
                         df_id,
                         nb_meta_data, 
                         df_code_cell,
                         df_md_cell,
                         max_n_code_cells,
                         max_n_md_cells,
                         is_train)
        self.mlm_probability = mlm_probability


    def __getitem__(self, index):
        nb_info = extract_nb_info(
            index, 
            self.df_id,
            self.nb_meta_data,
            self.df_code_cell,
            self.df_md_cell,
            self.max_n_code_cells,
            self.max_n_md_cells,
            self.is_train
        )

        # encode cells
        code_input_ids, code_attention_masks = encode_texts(
            nb_info["df_code_cell"], 
            nb_info["n_code_cell_pads"], 
            self.max_len, 
            self.front_lim, 
            self.back_lim, 
            self.ellipses_token_id,
            self.code_tokenizer,
            for_mlm=True,
            return_special_tokens_mask=False
        )
        md_input_ids, md_attention_masks, md_special_tokens_masks = encode_texts(
            nb_info["df_md_cell"], 
            nb_info["n_md_cell_pads"], 
            self.max_len, 
            self.front_lim, 
            self.back_lim, 
            self.ellipses_token_id,
            self.md_tokenizer,
            for_mlm=True,
            return_special_tokens_mask=True
        )

        # mask md cells
        masked_md_input_ids, masked_md_labels = mask_tokens(
            md_input_ids, 
            md_special_tokens_masks, 
            self.md_tokenizer, 
            self.mlm_probability
        )

        # cell attention masks
        code_cell_padding_masks = torch.zeros(nb_info["max_n_code_cells"]).bool()
        code_cell_padding_masks[nb_info["n_code_cells"]:] = True
        md_cell_padding_masks = torch.zeros(nb_info["max_n_md_cells"]).bool()
        md_cell_padding_masks[nb_info["n_md_cells"]:] = True

        return {
            'nb_id': nb_info["nb_id"],
            'code_input_ids': code_input_ids, 
            'code_attention_masks': code_attention_masks,
            'masked_md_input_ids': masked_md_input_ids,
            'md_attention_masks': md_attention_masks,
            'code_cell_padding_masks': code_cell_padding_masks,
            'md_cell_padding_masks': md_cell_padding_masks,
            'masked_md_labels': masked_md_labels
        }
