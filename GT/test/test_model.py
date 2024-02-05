import unittest
from src.GT_dataset import GTDataset
from src.GT_model import CustomModel
import pandas as pd
from src.utils import *
from torch.utils.data import DataLoader


logger = get_logger(logger_conf=logging_conf)


class TestGeneralRecommender(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('test/data/train_ratings.csv')
        self.cfg = CFG('test/test.yaml')

    def test_GTmodel(self):
        dataset = GTDataset(self.df, self.cfg)
        self.cfg.cate_idx_len, self.cfg.node_idx_len, node_interaction, item_len = dataset.get_att()

        model = CustomModel(self.cfg, node_interaction).to(self.cfg.device)
        model.train()

        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        for data, target in loader:
            output, embedded_target = model(data, target)
            unittest.TestCase().assertEqual(list(output.shape), list(embedded_target.shape))
            


if __name__ == '__main__':
    unittest.main()