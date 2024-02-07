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
        self.cfg = CFG('test/test_inference.yaml')

    def test_GTmodel(self):
        dataset = GTDataset(self.df, self.cfg)
        self.cfg.cate_idx_len, self.cfg.node_idx_len, node_interaction, item_len, self.node_idx2obj = dataset.get_att()
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        model = CustomModel(self.cfg, node_interaction).to(self.cfg.device)

        model.eval()
        with torch.no_grad():
            for data, _ in loader:
                output, node_embedding = model(data)
              
                cosine_similarities = batch_cosine_similarity(node_embedding[-item_len:], output)
                closest_idx = torch.argmax(cosine_similarities, dim=-1)
                closest_idx = len(node_embedding) - item_len + closest_idx
                
                item_id = self.node_idx2obj[closest_idx]
                print(item_id)
                assert()

        assert()



if __name__ == '__main__':
    unittest.main()