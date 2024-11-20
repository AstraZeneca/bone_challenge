import json
import os
import random
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
import pandas as pd
import torch

from data_utils.dataset import BoneSlicesDataset
from training.validation_metrics import get_true_and_predicted_labels, get_predicted_labels

DEVICE = 'cuda' if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
MODEL_PATH = 'training/final_model/model_20240520_093559_14'
BATCH_SIZE = 64
NUM_WORKERS = 4
VALIDATION_FOLD_NUMBER = 0
TEST_CONFIG_PATH = 'data_utils/example_config.json'
OUTPUT_RESULTS_PATH = 'metadata_test/model_20240520_093559_14_predictions.csv'
config = json.load(open(TEST_CONFIG_PATH))

def load_model(model_weights_path, device):
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(512, 2)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.load_state_dict(torch.load(model_weights_path))
    resnet.to(device)
    
    return resnet
    
    
resnet = load_model(MODEL_PATH, DEVICE)
resnet.eval()
test_dataset = BoneSlicesDataset(json_config_filepath = TEST_CONFIG_PATH)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
predicted_labels = get_predicted_labels(resnet, test_dataloader, DEVICE)
result = test_dataset.metadata
result['predicted_labels'] = predicted_labels
result = result.sort_values(by=[config['slice_index_col'], config['image_name_col']])

output_results = []

for img in result.groupby(['Image Name']):
    image_name = img[0][0]
    predicted = np.asarray(img[1]['predicted_labels'])
    # applying morphological closing 
    predicted_filter = cv2.morphologyEx(predicted, cv2.MORPH_CLOSE, np.ones((5,1)))
    predicted_filter_index = (predicted_filter==0).argmax(axis=0)
    if config['growth_plate_index_col'] in img[1]:
        growth_plate_index = img[1][config['growth_plate_index_col']].tolist()[0]
        output_results.append({
        'Image Name': image_name,
        'Predicted Filter Index': predicted_filter_index.tolist()[0],
        'True Index' : growth_plate_index
        })
    else:
        output_results.append({
            'Image Name': image_name,
            'Predicted Filter Index': predicted_filter_index.tolist()[0]
        })
        
output_df = pd.DataFrame(output_results)

output_df.to_csv(OUTPUT_RESULTS_PATH, index=False)