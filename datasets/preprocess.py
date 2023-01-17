import random
import multiprocess as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
import os


SEED = 42
PICKLE_PROTOCOL = int(os.environ['PICKLE_PROTOCOL'])


def read_nb(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 
                'source': 'str'}
        )
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


def process_order(raw_data_dir):
    df_orders = pd.read_csv(raw_data_dir / 'train_orders.csv')
    df_orders['cell_order'] = df_orders['cell_order'].str.split()
    df_orders = df_orders.explode('cell_order')
    df_orders['rank'] = df_orders.groupby('id')['cell_order'].cumcount()
    df_orders['pct_rank'] = (
        df_orders['rank'] / df_orders.groupby('id')['cell_order'].transform('count')
    )
    df_orders.rename(columns={'cell_order': 'cell_id'}, inplace=True)
    return df_orders


def obtain_nb_info(df_merge, raw_data_dir):
    df_merge['n_code_cells'] = (df_merge['cell_type'] == 'code').astype(np.int8)
    df_merge['n_md_cells'] = (df_merge['cell_type'] == 'markdown').astype(np.int8)

    df_nb = df_merge.groupby('id', as_index=False).agg({
        'cell_id': 'count',
        'n_code_cells': 'sum',
        'n_md_cells': 'sum'
    }).rename(columns={'cell_id': 'n_cells'})
    df_nb['md_pct'] = df_nb['n_md_cells'] / df_nb['n_cells']
    df_ancestors = pd.read_csv(raw_data_dir / 'train_ancestors.csv', index_col='id')
    df_nb['ancestor_id'] = df_nb['id'].map(df_ancestors['ancestor_id'])

    # A dict for all notebook metadata
    nb_meta_data = df_nb.drop('ancestor_id', axis=1).set_index('id').to_dict(orient='index')
    for d in nb_meta_data.values():
        d['n_code_cells'] = int(d['n_code_cells'])
        d['n_md_cells'] = int(d['n_md_cells'])
    return df_nb, nb_meta_data


def filter_by_n_cells(df_ids, nb_meta_data, max_code_cells, max_md_cells):
    df = df_ids[
        (df_ids.map(lambda x: nb_meta_data[x]['n_code_cells']) <= max_code_cells) &
        (df_ids.map(lambda x: nb_meta_data[x]['n_md_cells']) <= max_md_cells)
    ].reset_index(drop=True)
    
    return df


def train_val_split(nb_meta_data, df_nb, val_size):
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    train_ind, val_ind = next(splitter.split(df_nb, groups=df_nb["ancestor_id"]))
    train_df = df_nb.loc[train_ind, 'id'].reset_index(drop=True)
    val_df = df_nb.loc[val_ind, 'id'].reset_index(drop=True)

    return train_df, val_df


def remove_test_ids(df, test_ids_path):
    """
    Filter out training notebooks that are in the test set.
    """
    
    test_ids = pd.read_pickle(test_ids_path)
    df = df[~df['id'].isin(test_ids.values)].reset_index(drop=True)
    return df


def remove_non_en_nbs(df, non_en_ids_path):
    """
    Filter out non-english notebooks.
    """
    
    non_en_ids = pd.read_pickle(non_en_ids_path)
    df = df[~df['id'].isin(non_en_ids.values)].reset_index(drop=True)
    return df


def preprocess(args):
    train_paths = list((args.raw_data_dir / 'train').glob('*.json'))
    train_paths = np.array(train_paths)
    
    random.seed(SEED)
    random.shuffle(train_paths)

    n_trains = int(len(train_paths) * args.train_size)
    train_paths = train_paths[args.start_train_idx:n_trains+args.start_train_idx]

    with mp.Pool(mp.cpu_count()) as p:
        notebooks_train = list(
            p.map(
                read_nb, 
                tqdm(train_paths, desc='Reading train notebooks')
            )
        )

    df = pd.concat(notebooks_train).reset_index()
    df = remove_test_ids(df, args.test_ids_path)
    df = remove_non_en_nbs(df, args.non_en_ids_path)

    df['is_code'] = (df['cell_type'] == 'code').astype(np.int8)
    df['pos'] = df.groupby('id')['cell_id'].cumcount() + 1  # [1:TOTAL_MAX_CELLS]
    # dummy start has 0 rel_pos, code cells have pos/n_code_cells => # last code cells, rel_pos = 1
    df['rel_pos'] = df['pos'] / df.groupby('id')['is_code'].transform('sum')
    df.loc[df['cell_type'] == 'markdown', 'rel_pos'] = 0.

    df_orders = process_order(args.raw_data_dir)
    df_merge = df.merge(df_orders, how='left', on=['id', 'cell_id'])
    df_merge = df_merge[[
        'id', 
        'cell_id', 
        'cell_type', 
        'is_code',
        'pos', 
        'rel_pos',
        'source', 
        'rank', 
        'pct_rank', 
    ]]

    df_nb, nb_meta_data = obtain_nb_info(df_merge, args.raw_data_dir)

    train_ids, val_ids = train_val_split(nb_meta_data, df_nb, args.val_size)
    train_ids = filter_by_n_cells(train_ids, nb_meta_data, args.max_n_code_cells, args.max_n_md_cells)
    
    train_ids.to_pickle(args.train_ids_path, protocol=PICKLE_PROTOCOL)
    val_ids.to_pickle(args.val_ids_path, protocol=PICKLE_PROTOCOL)
    df_merge[df_merge.cell_type == 'code'].to_pickle(args.df_code_cell_path, protocol=PICKLE_PROTOCOL)
    df_merge[df_merge.cell_type == 'markdown'].to_pickle(args.df_md_cell_path, protocol=PICKLE_PROTOCOL)
    json.dump(nb_meta_data, open(args.nb_meta_data_path, "wt"))

    del train_ids, val_ids, df_merge, nb_meta_data
