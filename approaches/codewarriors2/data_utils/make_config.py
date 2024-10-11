"""
A script to prepare a config file.
It describes the parameters for the custom dataset that inherit from torch.utils.data.Dataset 
"""

import json
CONFIG_FILE_NAME = 'config_binary_test.json'

parameters = {
    'x-view' : False,
    'y-view' : False,
    'z-view' : True,
    'data_dir' : 'sliced_data_test',
    'x_axis_dir_name' : 'x_axis',
    'y_axis_dir_name' : 'y_axis',
    'z_axis_dir_name' : 'z_axis',
    'metadata_dir' : 'metadata_test',
    'x_axis_csv' : None,
    'y_axis_csv' : None,
    'z_axis_csv' : 'test_zaxis.csv',
    'label_option' : 'is_before_growth_plate'
    }

with open(CONFIG_FILE_NAME, 'w') as f:
    json.dump(parameters, f)