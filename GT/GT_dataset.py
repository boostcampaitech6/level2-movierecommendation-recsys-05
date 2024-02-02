import torch
from torch.utils.data import Dataset
import pandas as pd
import random


class GTDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.df = self.df.sort_values(by=['user', 'time'])

        self.cate_len, self.node_len = _indexing_data(self.df)
        self.u_start_end = _u_start_end(self.df, cfg.min_seq)
        self.len = len(self.u_start_end)

        self.max_seq_len = cfg.seq_len
        self.node_cols = cfg.node_cols
        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.nodes = self.df[self.node_cols].values
        self.cate_features = self.df[self.cate_cols].values
        self.cont_features = self.df[self.cont_cols].values

        self.device = cfg.device


    def __getitem__(self, idx):
        start, end = self.u_start_end[idx]
        end += 1
        start = max(end - self.max_seq_len, start)
        seq_len = end - start

        target_idx = random.sample(range(start, end + 1), 10)

        with torch.device(self.device):
            # 0으로 채워진 output tensor 제작    
            node_query = torch.zeros(self.max_seq_len, 2, dtype=torch.long)             
            node_key = torch.zeros(self.max_seq_len, 2, dtype=torch.long)              
            cate_query = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
            cate_key = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
            cont_query = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
            cont_key = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
            mask = torch.BoolTensor(self.max_seq_len)
        
            # tensor에 값 채워넣기
            node_query[-seq_len:] = torch.ShortTensor(self.nodes[target_idx]) # 16bit signed integer
            node_key[-seq_len:] = torch.ShortTensor(self.nodes[start:end]) # 16bit signed integer
            cate_query[-seq_len:] = torch.ShortTensor(self.cate_features[target_idx]) # 16bit signed integer
            cate_key[-seq_len:] = torch.ShortTensor(self.cate_features[start:end]) # 16bit signed integer
            cont_query[-seq_len:] = torch.HalfTensor(self.cont_features[target_idx]) # 16bit float
            cont_key[-seq_len:] = torch.HalfTensor(self.cont_features[start:end]) # 16bit float
            mask[:-seq_len] = True
            mask[-seq_len:] = False        
                
            target = torch.FloatTensor()

        return {'node'          : node, 
                'cate_feature'  : cate_feature, 
                'cont_feature'  : cont_feature, 
                'mask'          : mask, 
                'target'        : target,
                }
    

    def _u_start_end(self, df: pd.DataFrame, min_seq: int) -> list:
        u_s_e = df.reset_index().groupby('user')['index']
        u_s_e = u_s_e.apply(lambda x: (x.min(), x.max()))
        u_s_e = [(min, i) for min, max in u_s_e for i in range(min + min_seq, max + 1)]

        return u_s_e
    

    def _indexing_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        
        def obj2idx(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, int]:
            # nan 값이 0이므로 위해 offset은 1에서 출발한다
            offset = 1

            # Transformer을 위한 categorical feature의 index를 구한다
            for col in self.cols:

                # 각 column마다 mapper를 만든다
                obj2idx = {}
                for v in df[cols].unique():

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