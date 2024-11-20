import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap


ANNOT_dict = {
    "SUBSTANCE": ['AIR', 'FAT', 'BONE_CANCELLOUS', 'BONE_CORTICAL', 'WATER'],
    "HU_min": [-3000, -120, 300, 500, 0],
    "HU_max": [-1000, -90, 400, 2000, 0], 
    'MASK_ANNOT': [1,2,3,4,5]
}

ANNOT = pd.DataFrame.from_dict(ANNOT_dict)
COLORS_ANNOT = ['black', 'gray', 'khaki', 'green', 'white', 'blue']
ANNOT_CMAP = ListedColormap(COLORS_ANNOT)

def create_annotation_mask(imgdata):
    mask = np.zeros_like(imgdata)
    for index, row in ANNOT.iterrows():
        annot_value = row['MASK_ANNOT']
        hu_min = row['HU_min']
        hu_max = row['HU_max']

        temp_mask = np.logical_and(imgdata >= hu_min, imgdata <= hu_max)
        mask[temp_mask] = annot_value
    
    return mask