import unittest
from src.GT_dataset import GTDataset
import pandas as pd
from src.utils import *


logger = get_logger(logger_conf=logging_conf)


class test_datasets(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('test/data/train_ratings.csv')
        self.cfg = CFG('test/test.yaml')

    def test_GTdataset(self):
        ds = GTDataset(self.df, self.cfg)
        item = ds[2]
        logger.info(item)
        assert()



if __name__ == '__main__':
    unittest.main()