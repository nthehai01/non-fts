import os
import numpy as np
import random
import torch
from typing import List
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd


MLM_PROBABILITY = float(os.environ['MLM_PROBABILITY'])
NON_MASKED_INDEX = int(os.environ['NON_MASKED_INDEX'])
STEMMER = WordNetLemmatizer()


def make_folder(folder: str):
    """
    Makes the folder if not already present
    Args:
        folder (str): Name of folder to create
    Returns:
        created (bool): Whether or not a folder was created
    """
    
    try:
        os.mkdir(folder)
        return True
    except FileExistsError:
        pass
    return False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_to_4sf(lr: List[float]) -> str:
    """Get string of lr list that is rounded to 4sf to not clutter pbar

    Warning:
        Doesn't work for floats > 10000
    """
    def _f(x) -> str:
        a = str(x).partition('e')
        return a[0][:5] + 'e' + a[-1]
    return '[' + ', '.join(map(_f, lr)) + ']'


def preprocess_text(text):
    text = str(text)

    text = text.lower().strip()

    # # Remove all the special characters
    # text = re.sub(r'\W', ' ', str(text))
    # text = text.replace('_', ' ')

    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # # Remove single characters from the start
    # document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # # Removing prefixed 'b'
    # document = re.sub(r'^b\s+', '', document)

    # # Lemmatization
    # tokens = document.split()
    # tokens = [STEMMER.lemmatize(word) for word in tokens]
    # tokens = [word for word in tokens if len(word) > 3]
    # document = ' '.join(tokens)

    text = text.replace("[SEP]", "")
    text = text.replace("[CLS]", "")
    # text = re.sub(' +', ' ', text)

    return text


def trunc_mid(ids, max_len, front_lim, back_lim, ellipses_token_id):
    """
    Truncate the middle part of the texts if it is too long
    Use a token (ellipses_token_id) to separate the front and back part
    """
    if len(ids) > max_len:
        return ids[:front_lim] + [int(ellipses_token_id)] + ids[-back_lim:]
    return ids


def encode_texts(df_cell, 
                 n_pads, 
                 max_len, 
                 front_lim, 
                 back_lim, 
                 ellipses_token_id,
                 tokenizer, 
                 for_mlm=False,
                 return_special_tokens_mask=False):
    if for_mlm:
        texts = (
            df_cell['source'].apply(preprocess_text).tolist() + 
            n_pads * ['padding' + tokenizer.sep_token]
        )  # len = max_n_cells + 2
    else:
        texts = (
            ['starting' + tokenizer.sep_token] +
            df_cell['source'].apply(preprocess_text).tolist() + 
            ['ending' + tokenizer.sep_token] +
            n_pads * ['padding' + tokenizer.sep_token]
        )  # len = max_n_cells + 2

    inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=False,
        return_special_tokens_mask=return_special_tokens_mask,
    )

    map_func = lambda ids: trunc_mid(ids, max_len, front_lim, back_lim, ellipses_token_id)
    tokens = list(map(map_func, inputs['input_ids']))
    tokens = torch.LongTensor(tokens)
    
    cell_masks = list(map(lambda x: x[:max_len], inputs['attention_mask']))
    cell_masks = torch.LongTensor(cell_masks)

    if return_special_tokens_mask:
        special_tokens_masks = list(map(lambda x: x[:max_len], inputs['special_tokens_mask']))
        special_tokens_masks = torch.LongTensor(special_tokens_masks)
        return tokens, cell_masks, special_tokens_masks
    
    return tokens, cell_masks


def extract_nb_info(nb_index, 
                    df_id,
                    nb_meta_data,
                    df_code_cell,
                    df_md_cell,
                    max_n_code_cells,
                    max_n_md_cells,
                    is_train):
    nb_id = df_id[nb_index]
    n_code_cells = nb_meta_data[nb_id]['n_code_cells']
    n_md_cells = nb_meta_data[nb_id]['n_md_cells']
    
    df_code_cell = df_code_cell.loc[nb_id].copy()
    df_code_cell = df_code_cell.to_frame().T \
        if type(df_code_cell) == pd.core.series.Series else df_code_cell
    df_md_cell = df_md_cell.loc[nb_id].copy()
    df_md_cell = df_md_cell.to_frame().T \
        if type(df_md_cell) == pd.core.series.Series else df_md_cell

    if is_train:
        # code cells
        n_code_cell_pads = int(max(0, max_n_code_cells - n_code_cells))
        max_n_code_cells = max_n_code_cells
        # md cells
        n_md_cell_pads = int(max(0, max_n_md_cells - n_md_cells))
        max_n_md_cells = max_n_md_cells
    else:
        # code cells
        n_code_cell_pads = 0
        max_n_code_cells = n_code_cells
        # md cells
        n_md_cell_pads = 0
        max_n_md_cells = n_md_cells

    return {
        "nb_id": nb_id, 
        "n_code_cells": n_code_cells, 
        "n_md_cells": n_md_cells, 
        "df_code_cell": df_code_cell, 
        "df_md_cell": df_md_cell, 
        "n_code_cell_pads": n_code_cell_pads,
        "n_md_cell_pads": n_md_cell_pads,
        "max_n_code_cells": max_n_code_cells,
        "max_n_md_cells": max_n_md_cells
    }


def mask_tokens(inputs, special_tokens_mask, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = NON_MASKED_INDEX  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & \
        masked_indices & \
        ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    
    return inputs, labels
    

def load_checkpoint(model, optimizer, scheduler, args):
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpoint['weights'], strict=False)

    if not args.restore_weights_only:
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optim_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        

def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any([layer in name for layer in layers_to_freeze]):
            param.requires_grad = False
            