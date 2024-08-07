import os
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import datetime
import socket
import json
import random

import evaluate

def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.
    
    Parameters:
    seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_checkpoint(run_id, model, optimizer, fold, epoch, score, loss, filename):
    """
    Create a checkpoint for model training.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used for training.
        epoch: The current epoch number.
        loss: The current loss value.
        filename: The filename to save the checkpoint.
    """
    checkpoint = {
        "run_id": run_id,
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_score": score,
        "best_val_loss": loss,
        # Add other necessary states here
    }
    torch.save(checkpoint, filename)


def update_loss(criterion, outputs, labels):
    loss = criterion(outputs.squeeze(), labels.squeeze().float())
    return loss

def calculate_scores(predictions, labels):
    predictions_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    predictions_np = [np.round(arr).astype(int).flatten() for arr in predictions_np]
    predictions_np = np.concatenate(predictions_np)

    scores = 0.0
    scores_squared = 0.0
    # Loop through each prediction and label pair in the batch
    for prediction, label in zip(predictions_np, labels_np):
        score = evaluate._calculate_score(prediction, label)
        scores += score
        scores_squared += score**2

    return scores, scores_squared

def load_checkpoint(checkpoint_file, model, optimizer):
    """
    Load model and optimizer state from a checkpoint file if it exists.

    Parameters:
    checkpoint_file (str): Path to the checkpoint file.
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
    int: The epoch to start training from.
    float: The best validation loss recorded.
    """
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print("Checkpoint loaded. Resuming training from epoch", start_epoch)
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        print("Checkpoint file not found. Starting training from scratch.")
    
    return start_epoch, best_val_loss

def validate_model(
    model, 
    val_data_loader, 
    device, 
    criterion
):
    """
    Validate the model on the validation dataset.

    Parameters:
    model (torch.nn.Module): The model to validate.
    val_data_loader (DataLoader): DataLoader for the validation dataset.
    device (torch.device): The device to run the model on.
    criterion (torch.nn.Module): The loss function.
    calculate_scores (function): Function to calculate the scores.

    Returns:
    float: Validation loss.
    float: Mean validation score.
    float: Standard deviation of the validation scores.
    """
    model.eval()
    val_running_loss = 0.0
    val_scores = 0.0
    val_scores_squared_sum = 0.0
    
    with torch.no_grad():
        for images, labels in val_data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            scores, scores_squared = calculate_scores(outputs, labels)
            outputs = outputs.squeeze()
            labels = labels.long()
            val_loss = criterion(outputs, labels.float())  # REGRESSION
            val_running_loss += val_loss.item() * images.size(0)
            val_scores += scores
            val_scores_squared_sum += scores_squared  # Add squared score for calculating standard deviation
    
    val_loss = val_running_loss / len(val_data_loader.dataset) 
    val_scores_mean = val_scores / len(val_data_loader.dataset)
    val_scores_variance = (val_scores_squared_sum / (len(val_data_loader.dataset))) - (val_scores_mean ** 2)
    val_scores_stdev = val_scores_variance ** 0.5
    
    return val_loss, val_scores_mean, val_scores_stdev

def save_best_model(
    model, 
    output_dir, 
    fold_nr,
    epoch, 
    best_val_score, 
    best_val_loss, 
    val_scores_mean, 
    val_loss, 
    best_model_path
):
    """
    Save the model if validation score improves.

    Parameters:
    model (torch.nn.Module): The model to save.
    output_dir (str): Directory to save the model.
    fold_nr (int): Fold number.
    best_val_score (float): Best validation score so far.
    best_val_loss (float): Best validation loss so far.
    val_scores_mean (float): Current validation mean score.
    val_loss (float): Current validation loss.
    best_model_path (str): Path to the previous best model.

    Returns:
    float: Updated best validation score.
    float: Updated best validation loss.
    str: Path to the saved best model.
    """
    if val_scores_mean > best_val_score or (abs(val_scores_mean - best_val_score) < 1e-6 and val_loss < best_val_loss):
        best_val_score = val_scores_mean
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if best_model_path:  # Delete previous best model
            os.remove(best_model_path)
        best_model_path = os.path.join(
            output_dir, f"fold_{fold_nr}_epoch_{epoch}_model_best_val_score_{best_val_score:.3f}_val_loss_{best_val_loss:.4f}.pt"
        )
        torch.save(model.state_dict(), best_model_path)
    
    return best_val_score, best_val_loss, best_model_path


def train_fold(
    run_id,
    device,
    model,
    criterion,
    optimizer,
    train_data_loader,
    val_data_loader,
    fold_nr,
    output_dir,
    writer,
    num_epochs,
    scaler=None,
):
    best_val_loss = float("inf")
    best_val_score = float("-inf")
    best_model_path = None

    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs), desc=f"Epochs - Fold {fold_nr}"):
        model.train()
        running_loss = 0.0
        train_scores = 0.0
        for images, labels in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:  # HALF training
                with autocast():
                    outputs = model(images.float())
                    loss = update_loss(criterion, outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images.float())  # Ensure input tensor is of type float
                loss = update_loss(criterion, outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            scores, _ = calculate_scores(outputs, labels)
            train_scores += scores

        train_loss = running_loss / len(train_data_loader.dataset)
        train_scores = train_scores / len(train_data_loader.dataset)
        
        # Validation loop
        val_loss, val_scores_mean, val_scores_stdev = validate_model(
            model, 
            val_data_loader, 
            device, 
            criterion
        )
        print(
            f"Fold {fold_nr} Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.5f}, Train Score: {train_scores:.3f}, Val Loss: {val_loss:.5f}, Val Score: {val_scores_mean:.3f}, Val Score std: {val_scores_stdev:.3f}"
        )

        best_val_score, best_val_loss, best_model_path = save_best_model(
            model, 
            output_dir, 
            fold_nr,
            epoch, 
            best_val_score, 
            best_val_loss, 
            val_scores_mean, 
            val_loss, 
            best_model_path
        )

        if epoch % 250 == 0 or epoch == (num_epochs - 1):
            create_checkpoint(
                run_id,
                model,
                optimizer,
                fold_nr,
                epoch,
                best_val_score,
                best_val_loss,
                filename=f"models/{run_id}/checkpoint_fold_{fold_nr}_epoch_{epoch}.pth",
            )

        # Save model for every epoch
        # model_filename = os.path.join(output_dir, f"fold_{fold_nr}_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
        # torch.save(model.state_dict(), model_filename)

        # Writing to TensorBoard
        writer.add_scalar(f"Fold_{fold_nr}/Loss/Train", train_loss, epoch + 1)
        writer.add_scalar(f"Fold_{fold_nr}/Score/Train", train_scores, epoch + 1)
        writer.add_scalar(f"Fold_{fold_nr}/Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar(f"Fold_{fold_nr}/Score/Validation", val_scores_mean, epoch + 1)
        writer.add_scalar(f"Fold_{fold_nr}/Score/Valitadion_Stdev", val_scores_stdev, epoch + 1)        
    # Close the SummaryWriter
    writer.close()

    return best_model_path

def predict_to_dataframe(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    
    predictions = []
    image_indices = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images.float())
            predictions.append(outputs.cpu())
            
            # Get the indices for this batch
            if isinstance(dataloader.sampler, torch.utils.data.SequentialSampler):
                # For SequentialSampler, we can calculate the indices
                batch_indices = list(range(len(image_indices), len(image_indices) + images.size(0)))
            elif hasattr(dataloader.sampler, 'indices'):
                # For samplers with an indices attribute (like RandomSampler)
                start_idx = len(image_indices)
                end_idx = start_idx + images.size(0)
                batch_indices = dataloader.sampler.indices[start_idx:end_idx]
            else:
                raise ValueError("Unsupported sampler type")
            
            image_indices.extend(batch_indices)

    predictions = torch.cat(predictions)

    # Now we need to map these indices back to image names
    dataset = dataloader.dataset
    image_names = [dataset.image_names[idx] for idx in image_indices]

    df = pd.DataFrame({
        "Image_Name": image_names, 
        "Prediction": np.round(predictions.squeeze(), 1).tolist()
    })
    
    return df

# Function to load the best model and generate predictions
def get_predictions(best_model_path, model, val_data_loader, device):
    # Load the best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    predictions_df = predict_to_dataframe(model, val_data_loader, device)
    return predictions_df

def save_config(run_id, args, print_screen=False):
    args_dict = vars(args)

    config_data = {
        "run_id": run_id,
        "datetime": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),       
        **args_dict
    }

    with open("run_history.json", mode="a") as file:
        json.dump(config_data, file)
        file.write("\n")

    if print_screen:
        print("Image Colaboratorium 2024 Training Script Run Configuration:")
        print(json.dumps(config_data, indent=4))
        print("")
        

def evaluate_and_save_predictions(
    model, 
    best_model_path, 
    val_dataloader_x, 
    val_dataloader_y, 
    device, 
    output_dir, 
    fold_index, 
    nr_val_eval
):
    """
    Load the model state, evaluate on validation datasets, and save predictions to CSV files.
    
    Parameters:
    model (torch.nn.Module): The model to evaluate.
    best_model_path (str): Path to the best model's state dictionary.
    val_dataset_x (Dataset): Validation dataset for axis 0.
    val_dataset_y (Dataset): Validation dataset for axis 1.
    device (torch.device): The device to run the model on.
    output_dir (str): Directory to save the prediction CSV files.
    fold_index (int): The fold index for naming the output files.
    nr_val_eval (int): Number of validation evaluations to perform.
    """
    
    info = "" if fold_index is None else f"_val_fold_{fold_index}"

    # Load the model state
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # Evaluate and save predictions
    for j in tqdm(range(nr_val_eval), "Evaluating..."):
        predictions_df0 = predict_to_dataframe(model, val_dataloader_x, device)
        predictions_df0.to_csv(f"{output_dir}/pred{info}_axis0_{j}.csv", index=False)
        
        predictions_df1 = predict_to_dataframe(model, val_dataloader_y, device)
        predictions_df1.to_csv(f"{output_dir}/pred{info}_axis1_{j}.csv", index=False)
