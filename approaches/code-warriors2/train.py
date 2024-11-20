"""
Script used to train the models
"""
from datetime import datetime
from time import time
import sys
import os

from multiprocessing import freeze_support
from pathlib import Path
from sklearn.metrics import precision_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
import torch

from training.validation_metrics import calculate_metric
from training.training_utils import train_one_epoch
from data_utils.dataset import BoneSlicesDataset


# Parameters
torch.manual_seed(0)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DEVICE = 'cuda' if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else 'cpu'
RESULTS_DIR = 'test_result_dir' # training results directory
WRITER_DIR = os.path.join(RESULTS_DIR, "logs")
MODEL_PATH = os.path.join(RESULTS_DIR, "saved_models")
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
EPOCHS = 20
METADATA_PATH = os.path.join('train.csv')
JSON_CONFIG_PATH = 'data_utils/config_binary_z_refactored.json' # path to config

print("--------------------")
print(f"TIMESTAMP: {TIMESTAMP}")
print(f"CURRENT WORKING DIR: {os.getcwd()}")
print(f"DEVICE: {DEVICE}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"WRITER_DIR: {WRITER_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_WORKERS: {NUM_WORKERS}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print(f"METADATA_PATH: {METADATA_PATH}")
print("--------------------")


# Augmentations
transforms = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomRotation(degrees=180),
    v2.ToDtype(torch.float32, scale=False)
])


# Prepare dataset
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

train_ds = BoneSlicesDataset(
    json_config_filepath=JSON_CONFIG_PATH, transform=transforms)
valid_ds = BoneSlicesDataset(json_config_filepath=JSON_CONFIG_PATH)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)


# Training
for it in range(1):
    print(f"\n##############################################")
    print(f"Iteration: {it + 1}")
    print(f"##############################################\n")

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(512, 2)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(resnet.parameters())

    writer = SummaryWriter(f'{WRITER_DIR}/Iteration_{it + 1}')
    epoch_number = 0
    best_vloss = sys.float_info.max
    start_time = time()

    if not os.path.exists(MODEL_PATH + f'/Iteration_{it + 1}'):
        os.makedirs(MODEL_PATH + f'/Iteration_{it + 1}')

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        epoch_start_time = time()

        resnet.train(True)
        avg_loss = train_one_epoch(
            epoch_number, writer, train_dl, optimizer, loss_fn, resnet, DEVICE, start_time)

        resnet.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vlabels = vdata
                voutputs = resnet(vinputs.to(DEVICE))
                vloss = loss_fn(voutputs, vlabels.to(DEVICE))
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print("#############################################################")
        print("Epoch results:")
        print(f'Loss train {avg_loss} valid loss: {avg_vloss}')
        validation_precision_score = calculate_metric(resnet, val_dl, device=DEVICE,
                                                      metric=lambda x, y: precision_score(x, y, average='macro'))
        print(
            f'Validation macro average precision: {validation_precision_score}')
        print(f'Epoch execution time {time() - epoch_start_time}')
        print("#############################################################\n\n")

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss}, epoch_number + 1)

        writer.add_scalars('Macro_averaged_precision_score',
                           {'Validation': validation_precision_score}, epoch_number + 1)

        writer.flush()

        best_vloss = avg_vloss
        model_path = f'model_{TIMESTAMP}_{epoch_number}'
        torch.save(resnet.state_dict(), os.path.join(MODEL_PATH, f"Iteration_{it + 1}", model_path))

        epoch_number += 1

    writer.close()
