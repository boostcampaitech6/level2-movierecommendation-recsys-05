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
        self.node_cols = cfg.node_cols
        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        self.target_len = cfg.target_len        
        
        self.cate_len, self.node_len = self._get_indexing_data(self.df)
        self.u_start_end = self._get_u_start_end(self.df, cfg.min_seq)
        self.len = len(self.u_start_end)

        self.nodes = self.df[self.node_cols].values
        self.cate_features = self.df[self.cate_cols].values
        self.cont_features = self.df[self.cont_cols].values

        self.device = cfg.device


    def __getitem__(self, idx):
        start, end = self.u_start_end[idx]
        end += 1
        start = max(end - self.max_seq_len, start)
        seq_len = end - start

        target_idx = np.random.choice(range(seq_len), 10, replace=False)
        target_idx = np.sort(target_idx)

        with torch.device(self.device):
            # 0으로 채워진 output tensor 제작    
            node_query = torch.zeros(self.max_seq_len, len(self.node_cols), dtype=torch.int32)        
            node_key = torch.zeros(self.max_seq_len, len(self.node_cols), dtype=torch.int32)        
            cate = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.int32)
            cont = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float16)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        
            # tensor에 값 채워넣기
            temp = torch.tensor(self.nodes[start:end], dtype=torch.int32)
            node_query[target_idx + self.max_seq_len - seq_len] = temp[target_idx]
            node_key[-seq_len:] = temp # 16bit signed integer
            node_key[target_idx + self.max_seq_len - seq_len] = 0

            cate[-seq_len:] = torch.tensor(self.cate_features[start:end], dtype=torch.int32) # 16bit signed integer
            cont[-seq_len:] = torch.tensor(self.cont_features[start:end], dtype=torch.float16) # 16bit float
            mask[:-seq_len] = True
            mask[-seq_len:] = False        

            ### target은 10개의 item을 랜덤으로 선택하여 넣는다
            target = torch.FloatTensor(self.nodes.iloc[target_idx, 1])

        return {'node'          : node_query, 
                'cate_feature'  : cate, 
                'cont_feature'  : cont, 
                'mask'          : mask, 
                'target'        : target,
                }
    

    def _get_u_start_end(self, df: pd.DataFrame, min_seq: int) -> list:
        u_s_e = df.reset_index().groupby('user')['index']
        u_s_e = u_s_e.apply(lambda x: (x.min(), x.max()))
        u_s_e = [(min, i) for min, max in u_s_e for i in range(min + min_seq, max + 1)]

        return u_s_e
    

    def _get_indexing_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        
        def obj2idx(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, int]:
            # nan 값이 0이므로 위해 offset은 1에서 출발한다
            offset = 1

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

                # mapper를 이용하여 index로 변경한다
                df[col] = df[col].map(obj2idx).astype(int)
                
                # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
                # 제외하기 위해 offset값을 지속적으로 추가한다
                offset += len(obj2idx)

            return offset
                
        
        cate_len = obj2idx(df, self.cate_cols)
        node_len = obj2idx(df, self.node_cols)

        return cate_len, node_len


    def get_att(self):
        return {'node'              : self.cate_len, 
                'node_len'          : self.node_len, 
                'user_start_end'    : self.u_start_end,
                }
        
        
    def __len__(self):
        return self.len