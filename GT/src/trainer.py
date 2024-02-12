import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from .model import GTModel
from .dataset import GTDataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm 

from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(cfg, node_interaction):
    model = GTModel(cfg, node_interaction)

    if cfg.inference:
        model_path = os.path.join(cfg.model_dir, cfg.model_name)
        model_state = torch.load(model_path)
        model.load_state_dict(model_state["model"])

    model = model.to(cfg.device)

    return model


def train(model: nn.Module, train_loader, optimizer: torch.optim.Optimizer, loss_fun, cfg):
    model.train()
    total_loss = 0.0

    for data, target in tqdm(train_loader, mininterval=1):
        
        optimizer.zero_grad()
        output, embedded_target = model(data, target)

        loss = loss_fun(output.view(-1, output.shape[-1]), embedded_target.view(-1, output.shape[-1]), torch.ones(1, device=cfg.device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    logger.info("TRAIN LOSS : %.4f", average_loss)

    return average_loss


def validate(model: nn.Module, valid_loader, loss_fun, cfg):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, target in tqdm(valid_loader, mininterval=1):
            output, embedded_target = model(data, target)

            loss = loss_fun(output.view(-1, output.shape[-1]), embedded_target.view(-1, output.shape[-1]), torch.ones(1, device=cfg.device))
            total_loss += loss.item()

    average_loss = total_loss / len(valid_loader)
    logger.info("TRAIN LOSS : %.4f", average_loss)

    return average_loss


def inference(cfg, model: nn.Module, prepared, output_dir: str):
    test_data = prepared['test_data']
    test_cfg = prepared['test_cfg']

    test_data = GTDataset(test_data, test_cfg)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        output_list = []
        for data in tqdm(test_loader, mininterval=1):
            output = model(data)
            output_list.append(output.cpu().detach().numpy())
        
    output_list = np.concatenate(output_list).flatten()

    logger.info("Saving Result ...")
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": output_list}).to_csv(path_or_buf=write_path, index_label="id")

    logger.info("Successfully saved submission as %s", write_path)


def get_data(cfg):
    DATADIR = cfg.data_dir
    
    if not os.path.exists(os.path.join(DATADIR, "GT_train.csv")):
        split_data(DATADIR)

    train = pd.read_csv(os.path.join(DATADIR, "GT_train.csv"))
    valid = pd.read_csv(os.path.join(DATADIR, "GT_valid.csv"))
    all_data = pd.read_csv(os.path.join(DATADIR, "train_ratings2.csv"))

    return train, valid, all_data


def split_data(DATADIR):
    data = pd.read_csv(os.path.join(DATADIR, "train_ratings2.csv"))
    ids = data['user'].unique()
    num_elements = len(ids)
    num_to_select = int(num_elements * 0.2)  # 20%

    # 무작위 인덱스 선택
    random_indices = np.random.choice(num_elements, size=num_to_select, replace=False)

    select = data['user'].apply(lambda x: True if x in ids[random_indices] else False)

    data[~select].to_csv(os.path.join(DATADIR, "GT_train.csv"), index=False)
    data[select].to_csv(os.path.join(DATADIR, "GT_valid.csv"), index=False)
    data.to_csv(os.path.join(DATADIR, "GT_inference.csv"), index=False)