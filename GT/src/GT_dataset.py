import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np


class GTDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.df = self.df.sort_values(by=['user', 'time'])

        self.max_seq_len = cfg.seq_len
        self.target_len = cfg.target_len 
        self.device = cfg.device

        ### 각 종류의 feature들의 이름을 저장한다
        self.node_col_names = cfg.node_col_names
        self.cate_col_names = cfg.cate_col_names
        
        self.cate_idx_len, self.node_idx_len, node_idx2obj = self._get_indexing_data(self.df)
        self.node_idx2obj = torch.tensor(node_idx2obj, dtype=torch.int64, device=self.cfg.device)
        self.item_len = len(df['item'].unique())
        self.node_interaction = self.get_node_interaction(self.df, cfg.node_col_names)
        self.u_start_end = self._get_u_start_end(self.df, cfg.min_seq)
        self.len = len(self.u_start_end)

        ### 각 종류의 feature들에 대응하는 query, key를 만들기 위한 정보를 저장한다
        self.ingredients = {
            'names'       : ('node',                                'cate',                             'cont'),
            'dtypes'      : (torch.int32,                           torch.int32,                        torch.float32),
            'defaults'    : ((0, 1),                                1,                                  1.0),
            'lengths'     : (len(cfg.node_col_names),               len(cfg.cate_col_names),            len(cfg.cont_col_names)),
        }
    

    def __getitem__(self, idx):
        start, end = self.u_start_end[idx]
        end += 1
        start = max(end - self.max_seq_len, start)
        seq_len = end - start

        ### target은 self.target_len개의 item을 랜덤으로 선택한다.
        target_idx = np.random.choice(range(seq_len), self.target_len, replace=False)
        target_idx = np.sort(target_idx)

        query_idx = target_idx + self.max_seq_len - seq_len

        output = {}

        with torch.device(self.device):
            data = self.df.values
            data = torch.tensor(data[start:end], dtype=dtype)

            if self.infer is None:
                pass
            else:
                query_idx = query_idx[::-1]
                for index in query_idx:
                    temp = torch.cat((temp[:index], default, temp[index:]), 0)
            
            for name, dtype, default, length in zip(*[self.ingredients[key] for key in self.ingredients.keys()]):
                seq = torch.zeros(self.max_seq_len, length, dtype=dtype)
                seq[-seq_len:] = data[:,length]
                seq[query_idx] = torch.tensor(default, dtype=dtype)

                output[f'{name}'] = seq
                    
            output['query_idx'] = torch.tensor(query_idx, dtype=torch.int64)

            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            mask[query_idx] = False  
            output['mask'] = mask

            target = torch.tensor(self.df[self.node_col_names].values[target_idx, 1], dtype=torch.int32)

        return output, target


    def _get_indexing_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        
        def obj2idx(df: pd.DataFrame, cols: list, offset: int = 1) -> tuple[pd.DataFrame, int]:
            idx2obj = [-1 for _ in range(offset)]

            # Transformer을 위한 categorical feature의 index를 구한다
            for col in cols:

                # 각 column마다 mapper를 만든다
                obj2idx = {}
                for v in df[col].unique():

                    # np.nan != np.nan은 True가 나온다
                    # nan 및 None은 넘기는 코드
                    if (v != v) | (v == None):
                        continue 

                    # nan을 고려하여 offset을 추가한다
                    obj2idx[v] = len(obj2idx) + offset
                    idx2obj.append(v)

                # mapper를 이용하여 index로 변경한다
                df[col] = df[col].map(obj2idx).astype(int)
                
                # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
                # 제외하기 위해 offset값을 지속적으로 추가한다
                offset += len(obj2idx)

            return offset, idx2obj
                
        ### nan 값은 0으로, dummy cate는 1 나머지는 2부터 시작하도록 한다
        cate_idx_len, _ = obj2idx(df, self.cate_col_names, offset=2)

        ### nan 값은 0, dummy user는 1, dummy item은 2로 나머지는 3부터 시작하도록 한다
        node_idx_len, node_idx2obj = obj2idx(df, self.node_col_names, offset=3)

        return cate_idx_len, node_idx_len, node_idx2obj
    

    def _get_u_start_end(self, df: pd.DataFrame, min_seq: int) -> list:
        u_s_e = df.reset_index().groupby('user')['index']
        u_s_e = u_s_e.apply(lambda x: (x.min(), x.max()))
        u_s_e = [(min, i) for min, max in u_s_e for i in range(min + min_seq, max + 1)]

        return u_s_e


    def get_node_interaction(self, df, node_col_names):
        node_interaction = df[node_col_names].transpose().values
        node_interaction = torch.tensor(node_interaction, dtype=torch.int64).to(self.device)

        return node_interaction


    def get_att(self):
        return self.cate_idx_len, self.node_idx_len, self.node_interaction, self.item_len, self.node_idx2obj
        

    def __len__(self):
        return self.len