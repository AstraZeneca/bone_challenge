import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from src.models import BoneNet3d_2H
from src.utils import get_predictions
from src.dataset_3dwindow import ValidDataset
from glob import glob
from argparse import ArgumentParser


def do_predictions(checkpoint, device, config):
    test_annot = pd.read_csv(config['annotation_file'])
    model = BoneNet3d_2H(pretrained_weights=checkpoint, pretrained_weights_backbone=None)
    valid_dataset = ValidDataset(test_annot, config['img_dir'], config['seq_len'], config['stride'])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['batch_size'], num_workers=1)
    model.to(device)
    seqlen = valid_dataset.seq_len
    full_output = get_predictions(model, valid_loader, seqlen, device, to_int=False)
    return full_output

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                    help="config file for inference")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = glob(f'{config["checkpoints_dir"]}/*/model_best.pth.tar')

    outputs_final = []
    for checkpoint in checkpoints:
        outputs_final.append(do_predictions(checkpoint, device, config))
    preds = {}
    for imgname in outputs_final[0].keys():
        if imgname:
            preds[imgname] = [v[imgname]['pred_zindx'] for v in outputs_final]    
    os.makedirs(config["output_folder"], exist_ok=True)
    with open(f'{config["output_folder"]}/output_CV.json', 'w') as f:
        json.dump(preds, f)

    preds_mean = {k:int(np.round(np.mean(val))) for k, val in preds.items()}
    pred_df = pd.DataFrame(dict(preds_mean).items())
    pred_df .columns = ['Image Name', 'Zpred']
    pred_df['Image Name'] = pred_df['Image Name'].str.replace('.nii', '')
    pred_df.to_csv(f'{config["output_folder"]}/PredictedZ.csv')
