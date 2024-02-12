import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def get_logger(logger_conf: dict = logging_conf):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


class CFG:
    def __init__(self, config_file: str):
        extension = config_file.split(sep='.')[-1]
        if extension == 'json':
            import json
            form = json
            with open(config_file, 'r') as f:
                config = form.load(f)

        elif extension == 'yaml':
            import yaml
            form = yaml
            with open(config_file, 'r') as f:
                config = form.load(f, Loader=yaml.FullLoader)

        else:
            raise TypeError

        for key, value in config.items():
            setattr(self, key, value)


class CosineScalarLoss(nn.Module):
    def __init__(self, cfg):
        super(CosineScalarLoss, self).__init__()
        self.l2_param = cfg.hidden_size
        self.l2_param = self.l2_param ** (1/2)
        self.cos = nn.CosineEmbeddingLoss()


    def forward(self, input1, input2, target):
        ### cosine similarity를 이용한 loss
        cos_loss = self.cos(input1, input2, target)

        ### L2 norm을 이용한 loss
        norm_a = torch.norm(input1, p=2)
        norm_b = torch.norm(input2, p=2)
        absub = torch.abs(norm_a - norm_b)
        l2_loss = absub / (absub + self.l2_param)

        return cos_loss + l2_loss


def batch_cosine_similarity(embedding_matrix, vectors):
    """
    embedding_matrix와 vectors 간의 코사인 유사도를 배치로 계산합니다.
    
    Args:
    - embedding_matrix (torch.Tensor): 임베딩 테이블 행렬, 크기는 (num_embeddings, embedding_dim)
    - vectors (torch.Tensor): 비교할 벡터들의 배치, 크기는 (num_vectors, embedding_dim)
    
    Returns:
    - torch.Tensor: 코사인 유사도 행렬, 크기는 (num_vectors, num_embeddings)
    """
    # 내적을 계산하기 전에 두 행렬을 정규화합니다.
    embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
    vectors_norm = vectors / vectors.norm(dim=-1, keepdim=True)
    
    # 정규화된 행렬의 내적을 계산하여 코사인 유사도를 얻습니다.
    cosine_similarity = torch.matmul(vectors_norm, embedding_matrix_norm.t())
    
    return cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", type=bool, help="Use GPU")
    parser.add_argument("--data_dir", type=str, help="")
    parser.add_argument("--output_dir", type=str, help="")
    parser.add_argument("--model_dir", type=str, help="")
    parser.add_argument("--model_name", type=str, help="")
    
    parser.add_argument("--batch_size", type=int, help="")
    parser.add_argument("--emb_size", type=int, help="")
    parser.add_argument("--hidden_size", type=int, help="")
    parser.add_argument("--n_layers", type=int, help="")
    parser.add_argument("--n_head", type=int, help="")
    parser.add_argument("--seq_len", type=int, help="")
    parser.add_argument("--n_epochs", type=int, help="")
    parser.add_argument("--lr", type=float, help="")
    parser.add_argument("--dropout", type=float, help="")

    parser.add_argument("--verbose", type=bool, help="")
    

    args = parser.parse_args()

    return args


def parse_cfg(file: str) -> CFG:
    args = parse_args()
    cfg = CFG(file)

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    return cfg


def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
