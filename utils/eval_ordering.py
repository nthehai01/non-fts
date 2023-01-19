import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc


def get_raw_preds(model: nn.Module, loader: DataLoader, reg_criterion, device, name, mode):
    model.eval()
    pbar = tqdm(loader, desc=name)    
    nb_ids = []
    point_preds = []
    point_loss_list = []
    with torch.inference_mode():
        for batch in pbar:
            nb_ids.extend(batch['nb_id'])
            
            for attr in batch:
                if attr != 'nb_id':
                    batch[attr] = batch[attr].to(device)
            
            with torch.cuda.amp.autocast(False):
                point_pred = model(
                    batch['code_input_ids'],
                    batch['code_attention_masks'],
                    batch['md_input_ids'],
                    batch['md_attention_masks'],
                    batch['code_cell_padding_masks'],
                    batch['md_cell_padding_masks'],
                    batch['md_pct']
                )

            indices = torch.where(batch['reg_masks'] == True)
            point_preds.extend(point_pred[indices].cpu().numpy().tolist())

            if mode == "eval":
                reg_mask = batch['reg_masks'].float()
                point_loss = reg_criterion(
                    point_pred*reg_mask, 
                    batch['point_pct_target']
                ) * batch['n_md_cells']
                point_loss = point_loss.sum() / (batch['n_md_cells']*reg_mask).sum()
                # point_loss = point_loss.mean()

                point_loss_list.append(point_loss.item())
        
    # tidy up
    del point_pred
    gc.collect()
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
   
    return point_preds, point_loss_list
    

def get_point_preds(point_preds: np.array, df: pd.DataFrame, mode):
    df = df.reset_index()
    df.loc[df.cell_type == "markdown", 'rel_pos'] = point_preds
    df['pred_rank'] = df.groupby('id')['rel_pos'].rank()
    code_rank_correction(df)

    if mode == "eval":
        return df.sort_values('pp_rank').groupby('id')['cell_id'].apply(list)
    if mode == "test":
        res = df.sort_values('pp_rank').groupby('id')['cell_id'].apply(lambda x: " ".join(x)).reset_index()
        res.rename(columns={"cell_id": "cell_order"}, inplace=True)
        return res


def code_rank_correction(df):
    """Swap the code cells based on the given order
    """
    df['pp_rank'] = df['pred_rank'].copy()
    df.loc[df['cell_type'] == 'code', 'pp_rank'] = df.loc[
        df['cell_type'] == 'code'
    ].sort_values(['id', 'rel_pos'])['pred_rank'].values
    print('> Non-corrected %:', (df['pp_rank'] == df['pred_rank']).mean())


def predict(model, loader, df, device, name, mode="test", reg_criterion=None):
    assert mode in ["eval", "test"]

    preds, val_loss_list = get_raw_preds(model, loader, reg_criterion, device, name, mode)
    pred_series = get_point_preds(preds, df, mode)

    return pred_series, val_loss_list
