import torch
from torch.utils.data.dataloader import DataLoader
from src.models import BoneNet3d_2H
from src.utils import Trainer_sequence
import pandas as pd
from src.dataset_3dwindow import TrainDataset, ValidDataset
import numpy as np
import random 
from datetime import datetime
import os
import json
from argparse import ArgumentParser

def worker_init_fn(worker_id):                                                                                                                                
    seed = torch.utils.data.get_worker_info().seed % 2**32                                                                                                           
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    return

def train(train_annot, valid_annot, config):
    date = datetime.today().strftime('%Y-%m-%d-%H-%M')
    out_folder = f'{config["output_folder"]}/{date}_CV_paper'
    os.makedirs(out_folder, exist_ok=True)
    with open(f'{out_folder}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    batch_size = config['batch_size']   
    seq_len = config['seq_len']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = BoneNet3d_2H(pretrained_weights=config['pretrained_weights'], pretrained_weights_backbone=config['pretrained_weights_backbone'], seq_len = seq_len//2)

    valid_dataset = ValidDataset(valid_annot, config['img_dir'], seq_len=seq_len, stride=config['valid_stride'])
    valid_train_dataset = ValidDataset(train_annot, config['img_dir'], seq_len=seq_len, stride=config['valid_stride'])
    train_dataset = TrainDataset(train_annot, config['img_dir'], seq_len=seq_len)
        
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=2)
    valid_train_loader = DataLoader(dataset=valid_train_dataset, batch_size=batch_size, num_workers=2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=2, worker_init_fn=worker_init_fn)
    
    trainer = Trainer_sequence(config, out_folder, model, train_loader, valid_loader, valid_train_loader, device)
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                    help="config file for training")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    #sequentially do CV
    annot = pd.read_csv(config['annotation_file'])
    folds = annot['fold'].unique()
    for fold_id in folds:
        print (f'### FOLD {fold_id} ####')
        train_annot = annot[annot.fold!=fold_id].reset_index(drop=True).copy()
        valid_annot = annot[annot.fold==fold_id].reset_index(drop=True).copy()
        train(train_annot, valid_annot, config)