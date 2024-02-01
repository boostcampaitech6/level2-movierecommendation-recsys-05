import os
import unittest

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestGeneralRecommender(unittest.TestCase):
    def test_lightgcn(self):
        config_dict = {
            'model': 'LightGCN',
        }
        quick_test(config_dict)

    def test_transformer(self):
        config_dict = {
            'model': 'TransformerModel',
        }
        quick_test(config_dict)

    def test_custom_model(self):
        config_dict = {
            'model': 'CustomModel',
        }
        quick_test(config_dict)


if __name__ == '__main__':
    unittest.main()