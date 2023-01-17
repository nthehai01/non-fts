import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import warnings
import gc

from utils import load_checkpoint, make_folder, seed_everything
from datasets import get_dataloader
from models.notebook_mlm import NotebookMLM
from datasets.preprocess import preprocess
from utils.eval_mlm import predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


TRAIN_MODE = "mlm"
SEED = int(os.environ['SEED'])
NON_MASKED_INDEX = int(os.environ['NON_MASKED_INDEX'])
MAX_GRADIENT = float(os.environ['MAX_GRADIENT'])


def train_one_epoch(model, 
                    train_loader, 
                    criterion, 
                    optimizer, 
                    scheduler, 
                    state_dicts, 
                    epoch, 
                    args):
    loss_list = []
    model.train()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    for idx, batch in enumerate(pbar):
        for attr in batch:
            if attr != 'nb_id':
                batch[attr] = batch[attr].to(args.device)
        
        with torch.cuda.amp.autocast():
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

        masked_lm_loss.backward()
        loss_list.append(masked_lm_loss.item())

        # Clip the norm of the gradients to MAX_GRADIENT (may be 10.0)
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT)
        if idx % args.accumulation_steps == 0 or idx == len(pbar) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        step = 20
        if idx % step == 0:
            pbar.set_postfix(
                loss=np.round(np.mean(loss_list[-step:]), 4), 
                lr=scheduler.get_last_lr()[0]
            )

        if scheduler.get_last_lr()[0] == 0:
            print(" * lr=0, EARLY STOPPING")
            break

    state_dicts.update({
        'weights': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch
    })
    torch.save(state_dicts, f"{args.output_dir}/notebook_mlm_epoch{epoch}.tar")

    print("> Avg train Loss:", np.mean(loss_list))

    # tidy up
    del md_pred_scores
    gc.collect()
    if args.device == torch.device('cuda'):
        torch.cuda.empty_cache()

    return loss_list


def train(model, train_loader, val_loader, args):
    seed_everything(SEED)

    # creating optimizer and lr schedulers
    num_training_steps = len(train_loader) * args.epochs  # batches into the number of training epochs

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    # loading checkpoint
    if args.checkpoint_path is not None:
        load_checkpoint(model, optimizer, scheduler, args)
        print("Checkpoint loaded.")

    # logging
    state_dicts = {
        'weights': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': args.start_epoch
    }

    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=NON_MASKED_INDEX, reduction='mean')

    # # baseline
    # base_loss_list = predict(model, val_loader, criterion, args.device, "Get baseline")
    # print("> Avg base loss:", np.mean(base_loss_list))
    
    # training
    for epoch in range(args.start_epoch, args.epochs+1):
        _ = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            state_dicts, 
            epoch, 
            args
        )

        # # evaluate
        # val_loss_list = predict(model, val_loader, criterion, args.device, "Validating")
        # print("> Avg val loss:", np.mean(val_loss_list))

        if scheduler.get_last_lr()[0] == 0:
            break
        
        # REMOVE IF SUBMITTING
        args.n_epochs_per_save -= 1
        if args.n_epochs_per_save == 0:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')

    parser.add_argument('--code_pretrained', type=str, default="microsoft/codebert-base")    
    parser.add_argument('--md_pretrained', type=str, default="microsoft/codebert-base")    
    parser.add_argument('--n_heads', type=int, default=8)    
    parser.add_argument('--n_layers', type=int, default=6)   
    parser.add_argument('--max_n_code_cells', type=int, default=64)
    parser.add_argument('--max_n_md_cells', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--ellipses_token_id', type=int, default=734)

    parser.add_argument('--preprocess_data', action="store_true")
    parser.add_argument('--no-preprocess_data', action="store_false", dest='preprocess_data')
    parser.set_defaults(preprocess_data=True)
    parser.add_argument('--train_size', type=float, default=0.0001)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--start_train_idx', type=int, default=0)  # REMOVE IF SUBMITTING

    parser.add_argument('--test_ids_path', type=str, default="data/raw")
    parser.add_argument('--non_en_ids_path', type=str, default="data/raw")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    
    parser.add_argument('--n_epochs_per_save', type=int, default=100)  # REMOVE IF SUBMITTING

    parser.add_argument('--checkpoint_path', type=str, default=None)

    parser.add_argument('--output_dir', type=str, default="/Users/hainguyen/Documents/outputs")

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.raw_data_dir = Path(os.environ['RAW_DATA_DIR'])
    args.proc_data_dir = Path(os.environ['PROCESSED_DATA_DIR'])
    args.train_ids_path = args.proc_data_dir / "train_ids.pkl"
    args.val_ids_path = args.proc_data_dir / "val_ids.pkl"
    args.test_ids_path = Path(args.test_ids_path)
    args.non_en_ids_path = Path(args.non_en_ids_path)
    args.df_code_cell_path = args.proc_data_dir / "df_code_cell.pkl"
    args.df_md_cell_path = args.proc_data_dir / "df_md_cell.pkl"
    args.nb_meta_data_path = args.proc_data_dir / "nb_meta_data.json"

    args.restore_weights_only = False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args


if __name__ == '__main__':
    print("BIG NOTE: Don't forget to bash the env.sh!")
    print("="*50)

    args = parse_args()

    if args.preprocess_data:
        print("Preprocessing data...")
        preprocess(args)
        print("="*50)

    make_folder(args.output_dir)

    print("Loading training data...")
    train_loader = get_dataloader(args=args, objective=TRAIN_MODE, mode="train")
    print("Loading validating data...")
    val_loader = get_dataloader(args=args, objective=TRAIN_MODE, mode="val")
    print("="*50)
    
    model = NotebookMLM(
        args.code_pretrained, 
        args.md_pretrained, 
        args.n_heads, 
        args.n_layers
    )
    model.to(args.device)

    print("Starting training...")
    train(model, train_loader, val_loader, args)
