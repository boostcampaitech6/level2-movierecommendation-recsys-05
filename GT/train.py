import wandb
import copy
from torch.utils.data import DataLoader
import torch
from torch import nn
import os


from src.model import GTModel
from src.dataset import GTDataset
from src import trainer
from src.utils import get_logger, parse_cfg, set_seeds

logger = get_logger()


def main():
    cfg = parse_cfg('config_train.yaml')

    # wandb.login()
    # wandb.init(config=vars(cfg))
    set_seeds(cfg.seed)

    logger.info("Preparing data ...")
    train_data, valid_data, all_data = trainer.get_data(cfg)
    
    model_dir=cfg.model_dir
    os.makedirs(name=model_dir, exist_ok=True)

    all_data = GTDataset(all_data, cfg)
    all_data_cfg = copy.deepcopy(cfg)
    vars = all_data.get_att()
    all_data_cfg.node_idx_len, all_data_cfg.cate_idx_len = vars["node_idx_len"], vars["cate_idx_len"]
    node_idx2obj, cate_idx2obj = vars["node_idx2obj"], vars["cate_idx2obj"]
    all_data_node_interaction = vars["node_interaction"]

    train_data = GTDataset(train_data, cfg, node_idx2obj, cate_idx2obj)
    valid_data = GTDataset(valid_data, cfg, node_idx2obj, cate_idx2obj)
    
    model = trainer.build(all_data_cfg, all_data_node_interaction)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)

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
    main()
