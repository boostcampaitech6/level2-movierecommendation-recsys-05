import os
import unittest
import ..GT_dataset import GTDataset
import pandas as pd
from ..utils import CFG

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]



class test_dataset(unittest.TestCase):
    def test_dataset(self):
        df = pd.read_csv('data/train_ratings.csv')
        cfg = CFG('test_model.yaml')

        ds = GTDataset(df, cfg)

        for data in ds:
            print(data)



if __name__ == '__main__':
    unittest.main()