"""
A script to prepare a config file.
It describes the parameters for the custom dataset that inherit from torch.utils.data.Dataset 
"""

import json
CONFIG_FILE_NAME = 'example_config.json'

parameters = {
    '3d_images_dir' : '',
    'x_threshold_for_cropping' : 0.03,
    'y_threshold_for_cropping': 0.03,
    'slices_data_dir' : 'windowed_images_test',
    'original_metadata_path': 'train.csv', 
    'metadata_with_slices_path': 'metadata_test/metadata_with_slices_z_axis.csv',
    'label_option' : 'is_before_growth_plate',
    'slice_index_col' : 'Slice Index',
    'growth_plate_index_col' : 'Growth Plate Index',
    'image_name_col' : 'Image Name'
    }

with open(CONFIG_FILE_NAME, 'w') as f:
    json.dump(parameters, f)