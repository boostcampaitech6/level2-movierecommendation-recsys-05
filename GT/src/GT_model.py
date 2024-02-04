import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn.models import LightGCN



class GTModel(nn.Module):
    def __init__(self, cfg):
        super(GTModel, self).__init__()
        self.seq_len = cfg.seq_len
        emb_size, hidden_size = cfg.emb_size, cfg.hidden_size

        position_embedding = nn.Embedding(cfg.seq_len, emb_size)
        self.position_emb = position_embedding(torch.arange(cfg.seq_len, dtype=torch.long).unsqueeze(0))    ####???

        # node
        self.node_proj = nn.Sequential(
            nn.Linear(emb_size * len(cfg.node_col_names), hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # category
        self.cate_emb = nn.Embedding(cfg.cate_idx_len, emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(emb_size * len(cfg.cate_col_names), hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(len(cfg.cont_col_names))
        self.cont_proj = nn.Sequential(
            nn.Linear(len(cfg.cont_col_names), hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size*3, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.encoder = Encoder(cfg)

        self.reg_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(cfg.dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
    def forward(self, input: dict):
        batch_size = input['cate'].size(0)

        node_emb = self.node_proj(input['node'])
        
        # category
        cate_emb = self.cate_emb(input['cate']).view(batch_size, self.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(input['cont'].view(-1, input['cont'].size(-1))).view(batch_size, -1, input['cont'].size(-1))
        cont_emb = self.cont_proj(cont_x.view(batch_size, self.seq_len, -1))
        
        # combination
        seq_emb = torch.cat([node_emb, cate_emb, cont_emb], 2)
        seq_emb = self.comb_proj(seq_emb)

        seq_emb += self.position_emb

        encoded = self.encoder(seq_emb, input['query_index'], input['mask'])

        output = self.reg_layer(encoded)

        return output



class Encoder(nn.Module):
    def __init__(self, cfg, USE_BIAS=True):
        super(Encoder, self).__init__()
        if (cfg.hidden_size % cfg.n_head) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(cfg.hidden_size, cfg.n_head))
        self.d_feat = cfg.hidden_size
        self.n_head = cfg.n_head
        self.d_head = self.d_feat // self.n_head
        self.seq_len = cfg.seq_len
        self.sq_d_k = np.sqrt(self.d_head)
        self.dropout = nn.Dropout(p=cfg.dropout)

        self.lin_Q = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_K = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_V = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)

    def forward(self, seq, query_index, mask=None):
        n_batch = seq.shape[0]

        query = seq[query_index].contiguous()
        key = seq[np.setdiff1d(np.arange(self.seq_len), query_index)].contiguous()
        
        Q = self.lin_Q(query)
        K = self.lin_K(key)
        V = self.lin_V(key)

        Q = Q.view(n_batch, -1, self.n_head, self.d_head)
        K = K.view(n_batch, -1, self.n_head, self.d_head)
        V = V.view(n_batch, -1, self.n_head, self.d_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.sq_d_k 
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
            scores = scores.masked_fill(mask, -1e+9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attention), V) 

        output = output.transpose(1, 2).contiguous()
        output = output.view(n_batch, -1, self.d_feat)

        return output
    


class CustomModel(nn.Module):
    def __init__(self, cfg, node_interaction):
        super(CustomModel, self).__init__()
        LGCN = LightGCN(num_nodes=cfg.node_idx_len, embedding_dim=cfg.hidden_size, num_layers=cfg.hop).to(cfg.device)
        self.node_embedding = LGCN.get_embedding(node_interaction)

        self.GT = GTModel(cfg)

    def forward(self, input, target: None):
        node = self.node_embedding[input['node']]
        input['node'] = node.view(node.size(0), node.size(1), -1)

        output = self.GT(input)

        if target is None:
            return output
        else:
            target = self.node_embedding[target]
            
            return output, target
