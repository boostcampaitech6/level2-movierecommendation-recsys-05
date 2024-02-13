import wandb
import copy
from torch.utils.data import DataLoader
import torch
from torch import nn
import os
import multiprocessing as mp

from src.model import GTModel
from src.dataset import GTDataset
from src import trainer
from src.utils import *

logger = get_logger()


def main():
    cfg = parse_cfg('config_train.yaml')

    # wandb.login()
    # wandb.init(config=vars(cfg))
    set_seeds(cfg.seed)

    logger.info("Preparing data ...")
    train_data, valid_data, all_data = get_data(cfg)
    
    model_dir=cfg.model_dir
    os.makedirs(name=model_dir, exist_ok=True)

    ### nan 값은 0, dummy user는 1, dummy item은 2, 나머지는 3부터 시작하도록 한다
    cfg.node_idx_len, node_idx2obj, node_obj2idx = set_obj2idx(all_data, cfg.node_col_names, offset=3)
    
    ### nan 값은 0, dummy cate는 1, 나머지는 2부터 시작하도록 한다
    cfg.cate_idx_len, cate_idx2obj, cate_obj2idx = set_obj2idx(all_data, cfg.cate_col_names, offset=2)

    # node_idx2obj_tensor = torch.tensor(node_idx2obj, dtype=torch.int64, device=cfg.device)

    set_obj2idx(train_data, cfg.node_col_names, node_obj2idx, offset=3)
    set_obj2idx(train_data, cfg.cate_col_names, cate_obj2idx, offset=2)
    set_obj2idx(valid_data, cfg.node_col_names, node_obj2idx, offset=3)
    set_obj2idx(valid_data, cfg.cate_col_names, cate_obj2idx, offset=2)

    train_data = GTDataset(train_data, cfg, node_idx2obj, cate_idx2obj)
    valid_data = GTDataset(valid_data, cfg, node_idx2obj, cate_idx2obj)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True, num_workers=6)

    node_interaction = get_node_interaction(all_data, cfg)
    model = trainer.build(cfg, node_interaction)

    n_epochs=cfg.n_epochs
    learning_rate=cfg.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fun = nn.CosineEmbeddingLoss()

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_loss, best_epoch = -1, -1
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_loss = trainer.train(train_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun, cfg=cfg)
    
        # VALID
        valid_loss = trainer.validate(model=model, valid_loader=valid_loader, loss_fun=loss_fun, cfg=cfg)

        wandb.log(dict(train_loss_epoch=train_loss,
                       valid_loss_epoch=valid_loss,))

        if (valid_loss < best_loss) or (best_loss == -1):
            logger.info("Best model updated loss from %.4f to %.4f", best_loss, valid_loss)
            best_loss, best_epoch = valid_loss, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                       f=os.path.join(model_dir, f"best_model.pt"))
            
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1},
               f=os.path.join(model_dir, f"last_model.pt"))
    
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
