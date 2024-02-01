import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.models import LightGCN



class GCNTModel(nn.Module):
    def __init__(self, cfg):
        super(GCNTModel, self).__init__()
        self.seq_len = cfg.seq_len
        seq_len, emb_size, hidden_size = cfg.seq_len, cfg.emb_size, cfg.hidden_size

        total_col_size = cfg.cate_col_size + cfg.cont_col_size
        total_col_hidden_size = hidden_size * 2

        cate_col_hidden_size = int(cfg.cate_col_size / total_col_size * total_col_hidden_size)
        cont_col_hidden_size = total_col_hidden_size - cate_col_hidden_size


        position_embedding = nn.Embedding(seq_len, emb_size)
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        self.positions = position_embedding(positions)

        # category
        self.cate_emb = nn.Embedding(cfg.total_cate_size, emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(emb_size * cfg.cate_col_size, cate_col_hidden_size),
            nn.LayerNorm(cate_col_hidden_size),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(cfg.cont_col_size)
        self.cont_emb = nn.Sequential(
            nn.Linear(cfg.cont_col_size, cont_col_hidden_size),
            nn.LayerNorm(cont_col_hidden_size),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size*4, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.encoder = MultiHeadAttention(cfg)

        self.reg_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(cfg.dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, cfg.target_size),
        )
        
        
    def forward(self, node, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        # category
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_emb(cont_x.view(batch_size, self.seq_len, -1))
        
        # combination
        seq_emb = torch.cat([node, cate_emb, cont_emb], 2)
        seq_emb = self.comb_proj(seq_emb)

        seq_emb += self.positions

        encoded = self.encoder(seq_emb, mask)

        output = self.reg_layer(encoded)

        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, USE_BIAS=True):
        super(MultiHeadAttention, self).__init__()
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
        

    def forward(self, input, mask=None):
        n_batch = input.shape[0]
        
        Q = self.lin_Q(input)
        K = self.lin_K(input)
        V = self.lin_V(input)

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
    def __init__(self, node, cfg):
        super(CustomModel, self).__init__()
        self.node = node
        self.LGCN = LightGCN(num_nodes=cfg.node_size, embedding_dim=cfg.hidden_size, num_layers=cfg.hop)

        self.GCNT = GCNTModel(cfg)


    def forward(self, input, target: None):
        node_embedding = self.LGCN.get_embedding(self.node)

        node = node_embedding[input['node']]
        node = node.view(node.size(0), node.size(1), -1)

        output = self.GCNT(node, input["cate_feature"], input["cont_feature"], input["mask"])

        if target is None:
            return output
        else:
            target = node_embedding[target]
            
            return output, target
