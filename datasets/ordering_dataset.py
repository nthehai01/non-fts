import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import encode_texts, extract_nb_info


class OrderingDataset(Dataset):
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
                 is_train=False):
        super(OrderingDataset, self).__init__()
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_pretrained)
        self.md_tokenizer = AutoTokenizer.from_pretrained(md_pretrained)
        self.max_len = max_len
        self.front_lim = (max_len-2) // 2 + 2 - (max_len%2 == 0)
        self.back_lim = self.max_len - self.front_lim - 1
        self.ellipses_token_id = ellipses_token_id
        self.df_id = df_id
        self.nb_meta_data = nb_meta_data
        self.df_code_cell = df_code_cell
        self.df_md_cell = df_md_cell
        self.max_n_code_cells = max_n_code_cells
        self.max_n_md_cells = max_n_md_cells
        self.is_train = is_train


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
            for_mlm=False,
            return_special_tokens_mask=False
        )
        md_input_ids, md_attention_masks = encode_texts(
            nb_info["df_md_cell"], 
            nb_info["n_md_cell_pads"], 
            self.max_len, 
            self.front_lim, 
            self.back_lim, 
            self.ellipses_token_id,
            self.md_tokenizer,
            for_mlm=False,
            return_special_tokens_mask=False
        )

        # cell attention masks
        code_cell_padding_masks = torch.zeros(nb_info["max_n_code_cells"] + 2).bool()  # start + n_cells + end
        code_cell_padding_masks[nb_info["n_code_cells"]+2:] = True  # start to end are useful
        md_cell_padding_masks = torch.zeros(nb_info["max_n_md_cells"] + 2).bool()
        md_cell_padding_masks[nb_info["n_md_cells"]+2:] = True

        # n md cells
        n_md_cells_torch = torch.FloatTensor([nb_info["n_md_cells"]])

        # regression md masks
        reg_masks = torch.ones(nb_info["max_n_md_cells"]).bool()
        reg_masks[nb_info["n_md_cells"]:] = False

        # pointwise target for md cells
        point_pct_target = torch.FloatTensor(nb_info["df_md_cell"]['pct_rank'].tolist() + \
                           nb_info["n_md_cell_pads"]*[0.])

        # proportion of markdown cells
        if nb_info["n_md_cells"] + nb_info["n_code_cells"] == 0:
            md_pct = 0.
        else:
            md_pct = nb_info["n_md_cells"] / (nb_info["n_md_cells"] + nb_info["n_code_cells"])
        md_pct = torch.FloatTensor([md_pct])
        
        return {
            'nb_id': nb_info["nb_id"],
            'code_input_ids': code_input_ids, 
            'code_attention_masks': code_attention_masks,
            'md_input_ids': md_input_ids,
            'md_attention_masks': md_attention_masks,
            'code_cell_padding_masks': code_cell_padding_masks,
            'md_cell_padding_masks': md_cell_padding_masks,
            'n_md_cells': n_md_cells_torch,
            'reg_masks': reg_masks,
            'point_pct_target': point_pct_target,
            'md_pct': md_pct
        }


    def __len__(self):
        return len(self.df_id)
