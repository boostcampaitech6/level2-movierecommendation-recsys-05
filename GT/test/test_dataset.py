import os
import unittest
from src.GT_dataset import GTDataset
import pandas as pd
from src.utils import CFG



class test_dataset(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('test/data/train_ratings.csv')
        cfg = CFG('test/test_model.yaml')

        self.ds = GTDataset(df, cfg)

    def test_dataset(self):
        item = self.ds[2]
        print(item)
        self.assertEqual(item['node'].shape, (10, 3))



if __name__ == '__main__':
    unittest.main()