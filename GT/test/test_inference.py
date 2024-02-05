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
        self.cfg = CFG('test/test.yaml')

    def test_GTmodel(self):
        dataset = GTDataset(self.df, self.cfg)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.cfg.cate_idx_len, self.cfg.node_idx_len, node_interaction, item_len = dataset.get_att()

        model = CustomModel(self.cfg, node_interaction).to(self.cfg.device)

        model.eval()
        with torch.no_grad():
            for data in loader:
                output, node_embedding = model(data)
                
                distances = torch.norm(node_embedding[-item_len:] - output, dim=1) ### loss는 cosine으로 줄이고 inference는 euclidean으로 하는것은 이상하다
                closest_idx = torch.argmin(distances)             

        assert()
            


if __name__ == '__main__':
    unittest.main()