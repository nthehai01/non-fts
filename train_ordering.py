import argparse
from pathlib import Path
import os
import wandb
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import gc

from utils import load_checkpoint, freeze_layers, make_folder, seed_everything
from datasets import get_dataloader
from models.notebook_ordering import NotebookOrdering
from utils.metrics import kendall_tau
from utils.eval_ordering import predict
from datasets.preprocess import preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


TRAIN_MODE = "ordering"
SEED = int(os.environ['SEED'])


def train_one_epoch(model, 
                    train_loader, 
                    reg_criterion, 
                    scaler, 
                    optimizer, 
                    scheduler, 
                    state_dicts, 
                    epoch, 
                    args):
    point_loss_list = []
    model.train()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    for idx, batch in enumerate(pbar):
        for attr in batch:
            if attr != 'nb_id':
                batch[attr] = batch[attr].to(args.device)
        
        with torch.cuda.amp.autocast():
            point_pred = model(
                batch['code_input_ids'],
                batch['code_attention_masks'],
                batch['md_input_ids'],
                batch['md_attention_masks'],
                batch['code_cell_padding_masks'],
                batch['md_cell_padding_masks'],
                batch['md_pct'],
            )

            reg_mask = batch['reg_masks'].float()
            point_loss = reg_criterion(
                point_pred*reg_mask, 
                batch['point_pct_target']
            ) * batch['n_md_cells']
            # point_loss = point_loss.sum() / (batch['n_md_cells']*reg_mask).sum()
            point_loss = point_loss.mean()
            loss = point_loss

        scaler.scale(loss).backward()
        point_loss_list.append(point_loss.item())

        if idx % args.accumulation_steps == 0 or idx == len(pbar) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step = 1
        if idx % step == 0:
            metrics = {}
            metrics['point_loss'] = np.round(np.mean(point_loss_list[-step:]), 4)
            metrics['prev_lr'], metrics['next_lr'] = scheduler.get_last_lr()
            metrics['diff_lr'] = metrics['next_lr'] - metrics['prev_lr']
            wandb.log(metrics)

            pbar.set_postfix(
                loss=metrics['point_loss'], 
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
    torch.save(state_dicts, f"{args.output_dir}/notebook_ordering_epoch{epoch}.tar")

    metrics = {}
    metrics['avg_point_loss'] = np.mean(point_loss_list)
    wandb.log(metrics)
    print("> Avg train loss:", metrics['avg_point_loss'])

    # tidy up
    del point_pred, reg_mask
    gc.collect()
    if args.device == torch.device('cuda'):
        torch.cuda.empty_cache()

    return point_loss_list


def train(model, train_loader, val_loader, args):
    seed_everything(SEED)

    # creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    warmup_steps = len(train_loader) // args.accumulation_steps
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.lr,
        correct_bias=False  # To reproduce BertAdam specific behavior set correct_bias=False
    ) 
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_train_optimization_steps
    # )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_train_optimization_steps + args.epochs + 1,
        eta_min=args.lr / 5
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
    reg_criterion = torch.nn.L1Loss(reduction='none')
    scaler = torch.cuda.amp.GradScaler()

    # val baseline score
    val_ids = pd.read_pickle(args.val_ids_path)
    df_code_cell = pd.read_pickle(args.df_code_cell_path).set_index('id')
    df_md_cell = pd.read_pickle(args.df_md_cell_path).set_index('id')
    df_cell = df_code_cell.append(df_md_cell)
    val_df = df_cell.loc[val_ids.tolist()]
    df_orders = pd.read_csv(
        args.raw_data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    pred_series, val_loss_list = predict(
        model, 
        val_loader,  
        val_df, 
        args.device, 
        name="Get baseline",
        mode="eval", 
        reg_criterion=reg_criterion
    )
    print("> Avg baseline loss:", np.mean(val_loss_list))
    print('> Baseline score:', kendall_tau(df_orders.loc[pred_series.index], pred_series))

    # training
    for epoch in range(args.start_epoch, args.epochs+1):
        _ = train_one_epoch(
            model, 
            train_loader, 
            reg_criterion, 
            scaler, 
            optimizer, 
            scheduler, 
            state_dicts, 
            epoch, 
            args
        )

        # validation
        pred_series, val_loss_list = predict(
            model, 
            val_loader, 
            val_df, 
            args.device, 
            name="Validating",
            mode="eval",
            reg_criterion=reg_criterion
        )

        metrics = {}
        metrics['score'] = kendall_tau(df_orders.loc[pred_series.index], pred_series)
        wandb.log(metrics)
        print("> Avg val loss:", np.mean(val_loss_list))
        print("> Val score:", metrics['score'])

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

    parser.add_argument('--layers_to_freeze', nargs='+', type=str, default=[])

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
    parser.add_argument('--restore_weights_only', action="store_true")
    parser.add_argument('--no-restore_weights_only', action="store_false", dest='restore_weights_only')
    parser.set_defaults(restore_weights_only=True)

    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--wandb_name', type=str, default=None)
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

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


if __name__ == '__main__':
    print("BIG NOTE: Don't forget to bash the env.sh!")
    print("="*50)

    args = parse_args()
    wandb.init(
        project="AI4Code - Notebook Transformer", 
        name=args.wandb_name,
        mode=args.wandb_mode
    )

    try:
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
        
        model = NotebookOrdering(
            args.code_pretrained, 
            args.md_pretrained, 
            args.n_heads, 
            args.n_layers
        )
        model.to(args.device)

        wandb.watch(model, log_freq=10000, log_graph=True, log="all")

        print("Starting training...")
        if len(args.layers_to_freeze) > 0:
            freeze_layers(model, args.layers_to_freeze)
            print("{} layers freezed.".format(len(args.layers_to_freeze)))
        train(model, train_loader, val_loader, args)
    finally:
        wandb.finish()
