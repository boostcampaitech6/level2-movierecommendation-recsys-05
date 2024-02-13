import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np


class GTDataset(Dataset):
    def __init__(self, df, cfg, node_obj2idx=None, cate_obj2idx=None):
        self.df = df
        self.df = self.df.sort_values(by=['user', 'time'])

        self.max_seq_len = cfg.seq_len
        self.target_len = cfg.target_len 
        self.device = cfg.device
        self.inference = cfg.inference

        ### 각 종류의 feature들의 이름을 저장한다
        self.node_col_names = cfg.node_col_names
        self.cate_col_names = cfg.cate_col_names

        self.item_len = len(df['item'].unique())
        self.u_start_end = self._get_u_start_end(self.df, cfg.min_seq)
        self.len = len(self.u_start_end)

        ### 각 종류의 feature들에 대응하는 query, key를 만들기 위한 정보를 저장한다
        self.ingredients = {
            'names'       : ('node',                                'cate',                             'cont'),
            'dtypes'      : (torch.int32,                           torch.int32,                        torch.float32),
            'defaults'    : ((0, 1),                                [1] * len(cfg.cate_col_names),      [1.0] * len(cfg.cont_col_names)),
            'lengths'     : (len(cfg.node_col_names),               len(cfg.cate_col_names),            len(cfg.cont_col_names)),
            'datas'       : (self.df[cfg.node_col_names].values,    self.df[cfg.cate_col_names].values, self.df[cfg.cont_col_names].values),
        }
    

    def __getitem__(self, idx):
        start, end = self.u_start_end[idx]
        end += 1

        if self.inference:
            start = max(end - (self.max_seq_len - self.target_len), start)
            seq_len = end - start + self.target_len
            ### target은 self.target_len개의 item을 랜덤으로 선택한다.
            target_idx = np.random.choice(range(seq_len), self.target_len, replace=False)
            target_idx = np.sort(target_idx)
            query_idx = target_idx
            
        else:
            start = max(end - self.max_seq_len, start)
            seq_len = end - start
            target_idx = np.random.choice(range(seq_len), self.target_len, replace=False)
            target_idx = np.sort(target_idx)
            query_idx = target_idx + self.max_seq_len - seq_len

        output = {}

        with torch.device(self.device):
            for name, dtype, default, length, data in zip(*[self.ingredients[key] for key in self.ingredients.keys()]):
                data = torch.tensor(data[start:end], dtype=dtype)
                
                if self.inference:
                    for index in query_idx:
                        data = torch.cat((data[:index], data[0].unsqueeze(0), data[index:]), 0)

                seq = torch.zeros(self.max_seq_len, length, dtype=dtype)
                seq[-seq_len:] = data
                seq[query_idx] = torch.tensor(default, dtype=dtype)

                output[f'{name}'] = seq
                    
            output['query_idx'] = torch.tensor(query_idx, dtype=torch.int64)

            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            mask[query_idx] = False  
            output['mask'] = mask

            target = torch.tensor(self.df[self.node_col_names].values[target_idx, 1], dtype=torch.int32)

        return output, target


    def _get_u_start_end(self, df: pd.DataFrame, min_seq: int) -> list:
        u_s_e = df.reset_index().groupby('user')['index']
        u_s_e = u_s_e.apply(lambda x: (x.min(), x.max()))
        u_s_e = [(min, i) for min, max in u_s_e for i in range(min + min_seq, max + 1)]

        return u_s_e


    def get_item_len(self):
        return self.item_len
        

    def __len__(self):
        return self.len