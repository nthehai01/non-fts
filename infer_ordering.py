import argparse
from pathlib import Path
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import gc
import multiprocess as mp
import json

from utils import load_checkpoint, make_folder, seed_everything
from datasets import get_dataloader
from models.notebook_ordering import NotebookOrdering
from utils.eval_ordering import predict
from datasets.preprocess import read_nb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


TRAIN_MODE = "ordering"
SEED = int(os.environ['SEED'])
PICKLE_PROTOCOL = int(os.environ['PICKLE_PROTOCOL'])


def obtain_nb_info(df_merge, raw_data_dir):
    df_merge['n_code_cells'] = (df_merge['cell_type'] == 'code').astype(np.int8)
    df_merge['n_md_cells'] = (df_merge['cell_type'] == 'markdown').astype(np.int8)

    df_nb = df_merge.groupby('id', as_index=False).agg({
        'cell_id': 'count',
        'n_code_cells': 'sum',
        'n_md_cells': 'sum'
    }).rename(columns={'cell_id': 'n_cells'})
    df_nb['md_pct'] = df_nb['n_md_cells'] / df_nb['n_cells']

    # A dict for all notebook metadata
    nb_meta_data = df_nb.set_index('id').to_dict(orient='index')
    for d in nb_meta_data.values():
        d['n_code_cells'] = int(d['n_code_cells'])
        d['n_md_cells'] = int(d['n_md_cells'])
    return nb_meta_data


def dataset_preprocess(args):
    test_paths = list((args.raw_data_dir / 'test').glob('*.json'))
    test_paths = np.array(test_paths)
    
    with mp.Pool(mp.cpu_count()) as p:
        notebooks_test = list(
            p.map(
                read_nb, 
                tqdm(test_paths, desc='Reading test notebooks')
            )
        )

    df = pd.concat(notebooks_test).reset_index()

    df['is_code'] = (df['cell_type'] == 'code').astype(np.int8)
    df['pos'] = df.groupby('id')['cell_id'].cumcount() + 1  # [1:TOTAL_MAX_CELLS]
    # dummy start has 0 rel_pos, code cells have pos/n_code_cells => # last code cells, rel_pos = 1
    df['rel_pos'] = df['pos'] / df.groupby('id')['is_code'].transform('sum')
    df.loc[df['cell_type'] == 'markdown', 'rel_pos'] = 0.

    # dummy
    df["rank"] = df["pct_rank"] = 0

    df_merge = df[[
        'id', 
        'cell_id', 
        'cell_type', 
        'is_code',
        'pos', 
        'rel_pos',
        'source', 
        'rank', 
        'pct_rank'
    ]]

    nb_meta_data = obtain_nb_info(df_merge, args.raw_data_dir)

    test_ids = df_merge.id.unique()
    sorted_ids = np.sort(test_ids)
    test_ids = pd.Series(sorted_ids)

    test_ids.to_pickle(args.test_ids_path, protocol=PICKLE_PROTOCOL)
    df_merge[df_merge.cell_type == 'code'].to_pickle(args.df_code_cell_path, protocol=PICKLE_PROTOCOL)
    df_merge[df_merge.cell_type == 'markdown'].to_pickle(args.df_md_cell_path, protocol=PICKLE_PROTOCOL)
    json.dump(nb_meta_data, open(args.nb_meta_data_path, "wt"))

    del test_ids, df_merge, nb_meta_data


def infer(model, test_loader, args):
    seed_everything(SEED)
    
    # loading checkpoint
    if args.checkpoint_path is not None:
        load_checkpoint(model, None, None, args)
        print("Checkpoint loaded.")

    # val baseline score
    test_ids = pd.read_pickle(args.test_ids_path)
    df_code_cell = pd.read_pickle(args.df_code_cell_path).set_index('id')
    df_md_cell = pd.read_pickle(args.df_md_cell_path).set_index('id')
    df_cell = df_code_cell.append(df_md_cell)
    test_df = df_cell.loc[test_ids.tolist()]

    pred_series, _ = predict(
        model, 
        test_loader,  
        test_df, 
        args.device, 
        name="Testing",
        mode="test"
    )
    
    return pred_series


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')

    parser.add_argument('--code_pretrained', type=str, default="microsoft/codebert-base")    
    parser.add_argument('--md_pretrained', type=str, default="microsoft/codebert-base")    
    parser.add_argument('--n_heads', type=int, default=8)    
    parser.add_argument('--n_layers', type=int, default=6)   
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--ellipses_token_id', type=int, default=734)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="/Users/hainguyen/Documents/outputs")

    args = parser.parse_args()

    args.max_n_code_cells = 64
    args.max_n_md_cells = 64
    args.batch_size = 1

    args.output_dir = Path(args.output_dir)
    args.raw_data_dir = Path(os.environ['RAW_DATA_DIR'])
    args.proc_data_dir = Path(os.environ['PROCESSED_DATA_DIR'])
    args.test_ids_path = args.proc_data_dir / "test_ids.pkl"
    args.df_code_cell_path = args.proc_data_dir / "df_code_cell.pkl"
    args.df_md_cell_path = args.proc_data_dir / "df_md_cell.pkl"
    args.nb_meta_data_path = args.proc_data_dir / "nb_meta_data.json"
    args.submission_dir = args.output_dir / "submission.csv"
    args.restore_weights_only = True

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


if __name__ == '__main__':
    print("BIG NOTE: Don't forget to bash the env.sh!")
    print("="*50)

    args = parse_args()

    print("Preprocessing data...")
    dataset_preprocess(args)
    print("="*50)

    make_folder(args.output_dir)

    print("Loading testing data...")
    test_loader = get_dataloader(args=args, objective=TRAIN_MODE, mode="test")
    print("="*50)
    
    model = NotebookOrdering(
        args.code_pretrained, 
        args.md_pretrained, 
        args.n_heads, 
        args.n_layers
    )
    model.to(args.device)

    pred_series = infer(model, test_loader, args)

    pred_series.to_csv(args.submission_dir, index=False)
    