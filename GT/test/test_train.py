import unittest
from src.GT_dataset import GTDataset
from src.GT_model import CustomModel
import pandas as pd
from src.utils import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


logger = get_logger(logger_conf=logging_conf)


class TestGeneralRecommender(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('test/data/train_ratings.csv')
        self.cfg = CFG('test/test_model.yaml')

    def test_GTmodel(self):
        dataset = GTDataset(self.df, self.cfg)
        self.cfg.cate_idx_len, self.cfg.node_idx_len, node_interaction = dataset.get_att()

        model = CustomModel(self.cfg, node_interaction).to(self.cfg.device)
        model.train()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.cfg.lr)
        loss_fun = nn.CosineEmbeddingLoss()

        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        for data, target in loader:
            optimizer.zero_grad()
            output, embedded_target = model(data, target)

            loss = loss_fun(output.view(-1, output.shape[-1]), embedded_target.view(-1, output.shape[-1]), torch.ones(1, device=self.cfg.device))
            print(loss)
            loss.backward()
            optimizer.step()

        assert()
            


if __name__ == '__main__':
    unittest.main()