import unittest
from src.dataset import GTDataset
import pandas as pd
from src.utils import *


logger = get_logger(logger_conf=logging_conf)


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('test/data/train_ratings.csv')
        self.cfg = CFG('test/test_train.yaml')
        self.cfg.data_dir = 'test/data'

    def test_GTdataset(self):
        train_data, valid_data, all_data = get_data(self.cfg)
        self.cfg.node_idx_len, node_idx2obj, node_obj2idx = set_obj2idx(all_data, self.cfg.node_col_names, offset=3)
        self.cfg.cate_idx_len, cate_idx2obj, cate_obj2idx = set_obj2idx(all_data, self.cfg.cate_col_names, offset=2)

        set_obj2idx(train_data, self.cfg.node_col_names, node_obj2idx, offset=3)
        set_obj2idx(train_data, self.cfg.cate_col_names, cate_obj2idx, offset=2)
        set_obj2idx(valid_data, self.cfg.node_col_names, node_obj2idx, offset=3)
        set_obj2idx(valid_data, self.cfg.cate_col_names, cate_obj2idx, offset=2)

        train_data = GTDataset(train_data, self.cfg, node_idx2obj, cate_idx2obj)
        valid_data = GTDataset(valid_data, self.cfg, node_idx2obj, cate_idx2obj)
        item = train_data[2]
        print(item)
        assert()



if __name__ == '__main__':
    unittest.main()