import argparse
import glob
import os
import uuid
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import helpers
import datasets
import models

def main(args):
    # created a new run id and save the configuration
    run_id = str(uuid.uuid4())    
    helpers.save_config(run_id, args)

    # Constants definition    
    zaxis_size = 642
    img_sz = [models.model_input_sizes[args.model_type], zaxis_size]    
    loss_function = nn.MSELoss() # regression

    # Half training mode
    scaler = None
    if args.train_mode == "half":
        scaler = GradScaler()
            
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For reproducibility
    helpers.set_seed(args.seed)
        
    # Load the CSV file into a DataFrame
    labels_df = pd.read_csv(args.summary_path)

    # Extract the "Growth Plate Index" column as labels
    labels = labels_df["Growth Plate Index"].astype("float32").tolist()
    study_ids = labels_df["STUDY ID"].astype("int").tolist()

    # Assuming the "Image Name" column contains the file names without extensions
    image_names = labels_df["Image Name"].tolist()

    # Create a dictionary to map image names to their corresponding labels
    label_map = {
        image_name: label for image_name, 
        label in zip(image_names, labels)
    }

    # Use the label map to assign labels to file paths
    file_paths = glob.glob(args.nii_path)

    # Initialize a SummaryWriter for TensorBoard
    writer = SummaryWriter(f"logs/{run_id}")

    output_dir = f"models/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    stratified_kfold = StratifiedKFold(
        n_splits=args.num_folds, 
        shuffle=True, 
        random_state=args.seed
    )

    # Initialize lists to store training and validation results
    all_train_labels = []
    all_val_labels = []

    for fold, (train_index, val_index) in enumerate(stratified_kfold.split(file_paths, study_ids)):
        if args.fold is not None and args.fold != fold:
            continue # Skip the rest of the loop body and move to the next iteration
        train_file_paths = [file_paths[i] for i in train_index]
        val_file_paths = [file_paths[i] for i in val_index]

        # Extract the image names from the full paths
        train_image_names = [
            file_path.split("/")[-1][:-4] for file_path in train_file_paths
        ]
        val_image_names = [
            file_path.split("/")[-1][:-4] for file_path in val_file_paths
        ]

        # Extract labels for training and validation sets using the label map
        train_labels = [label_map[image_name] for image_name in train_image_names]
        val_labels = [label_map[image_name] for image_name in val_image_names]

        all_train_labels.append(train_labels)
        all_val_labels.append(val_labels)

        # Create datasets for training and validation
        train_data_loader, val_data_loader_xy = datasets.create_train_dataloaders(
            fold,
            train_file_paths,
            val_file_paths, 
            train_labels,
            val_labels, 
            img_sz, 
            args.central_prop, 
            args.central_prop_val, 
            args.zaxis_mode, 
            args.planes_index_selection, 
            args.num_input_channels,
            args.augmented_dataset_root_dir,
            args.use_existing_augmented_dir,   
            args.augmentation_train_count,
            args.augmentation_valid_count,
            args.batch_size,
            args.num_workers         
        )

        model = models.load_model(
            args.model_type, 
            num_input_channels=args.num_input_channels
        )
        model = model.to(device)
        criterion = loss_function.to(device)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr
        )

        best_model_path = helpers.train_fold(
            run_id,
            device,
            model,
            criterion,
            optimizer,
            train_data_loader,
            val_data_loader_xy,
            fold,
            output_dir,
            writer,
            args.epochs,
            scaler,
        )

        val_dataloader_x, val_dataloader_y = datasets.create_eval_dataloaders(
            val_file_paths, 
            val_labels, 
            img_sz, 
            args.central_prop_val, 
            args.planes_index_selection, 
            args.num_input_channels,
            args.batch_size,
            args.num_workers                 
        )

        helpers.evaluate_and_save_predictions(
            model, 
            best_model_path, 
            val_dataloader_x, 
            val_dataloader_y, 
            device, 
            output_dir, 
            fold, 
            args.nr_val_eval
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matterhorn Team - Workstream-2 Image Collaboratorium 2024 Training Script")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=None, help="Run only one fold. Specify the fold number.")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--central-prop", type=float, default=0.40, help="Central proportion")
    parser.add_argument("--central-prop-val", type=float, default=0.40, help="Central proportion for validation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--nr-val-eval", type=int, default=10, help="Number of validation evaluations per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model-type", choices=["B1", "B2", "B3", "B4", "B5", "B6", "B7"], default="B3", help="Model type")
    parser.add_argument("--summary-path", type=str, default="", help="Path to summary file")
    parser.add_argument("--nii-path", type=str, default="", help="Path to NII files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-mode", choices=["full", "half"], default="half", help="Training mode: full or half precision")
    parser.add_argument("--zaxis-mode", choices=["crop", "interpolate", "mixed"], default="mixed", help="z-Axis mode: Crop, Interpolate, Mixed") 
    parser.add_argument("--planes-index-selection", choices=["random", "consecutive"], default="random", help="")
    parser.add_argument("--num-input-channels", type=int, default=9, help="number of network input planes")
    parser.add_argument('--augmented-dataset-root-dir', type=str, default="", help='Specify the root directory for the augmented dataset.')        
    parser.add_argument('--use-existing-augmented-dir', action='store_true', help='Specify whether to use existing augmented datasets. If set, existing datasets will be used. If not, generate new.')
    parser.add_argument("--augmentation-train-count", type=int, default=150, help="number of augmentations per file in train")
    parser.add_argument("--augmentation-valid-count", type=int, default=10, help="number of augmentations per file in valid")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers of the DataLoader")

    args = parser.parse_args()
    main(args)