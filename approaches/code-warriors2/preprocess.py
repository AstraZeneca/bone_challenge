import json
import os

import pandas as pd

from niifiles_utils import window_and_slice_image_combined

CONFIG_PATH = os.path.join('data_utils', 'example_config.json')
config = json.load(open(CONFIG_PATH))

print('Config:\n', config)

def save_metadata_with_slices(config):
    
    flist = sorted(os.listdir(config['slices_data_dir']))
    flist = [f for f in flist if f.endswith('.npy')]
    print('Number of slices:', len(flist))
    df = pd.DataFrame()
    df['img_file_name'] = flist
    df[config['image_name_col']] = df['img_file_name'].apply(lambda x: x.split('.')[0])
    df[[config['image_name_col'], config['slice_index_col']]] = df[config['image_name_col']].str.split('_', expand=True)
    original_metadata_df = pd.read_csv(config['original_metadata_path'])
    df = df.merge(original_metadata_df, on = config['image_name_col'], how = 'left')
    if not os.path.exists(config['metadata_with_slices_path']):
        os.makedirs(os.path.dirname(config['metadata_with_slices_path']))
    df.to_csv(config['metadata_with_slices_path'])


window_and_slice_image_combined(config)

save_metadata_with_slices(config)