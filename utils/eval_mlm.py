import torch
from tqdm import tqdm
import gc


def predict(model, loader, criterion, device, name):
    model.eval()
    pbar = tqdm(loader, desc=name)    
    nb_ids = []
    loss_list = []
    with torch.inference_mode():
        for batch in pbar:
            nb_ids.extend(batch['nb_id'])
            
            for attr in batch:
                if attr != 'nb_id':
                    batch[attr] = batch[attr].to(device)
            
            with torch.cuda.amp.autocast(False):
                md_pred_scores = model(
                    batch['code_input_ids'],
                    batch['code_attention_masks'],
                    batch['masked_md_input_ids'],
                    batch['md_attention_masks'],
                    batch['code_cell_padding_masks'],
                    batch['md_cell_padding_masks']
                )

                vocab_size = md_pred_scores.shape[-1]
                masked_lm_loss = criterion(
                    md_pred_scores.view(-1, vocab_size),
                    batch['masked_md_labels'].view(-1)
                )

            loss_list.append(masked_lm_loss.item())

    # tidy up
    del md_pred_scores
    gc.collect()
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    return loss_list
