import wandb

from src import trainer
from src.utils import get_logger, parse_cfg, set_seeds

logger = get_logger()


def main():
    cfg = parse_cfg('config_train.yaml')

    wandb.login()
    wandb.init(config=vars(cfg))
    set_seeds(cfg.seed)

    logger.info("Preparing data ...")
    train, valid = trainer.split_data(cfg)

    logger.info("Building Model ...")
    model = trainer.build(cfg)
    
    logger.info("Start Training ...")
    trainer.run(model=model, train=train, valid=valid, cfg=cfg)



if __name__ == "__main__":
    main()
