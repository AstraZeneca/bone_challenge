import argparse
import glob
import os
import uuid
import torch

import helpers
import datasets
import models

def main(args):
    print("Warning: Prediction is done in crop mode. For full prediction capabilities, use the two-step process: first interpolated evaluation, then cropped evaluation.")
    print("Pending implementation of the two step method.")
    
    # created a new run id and save the configuration
    run_id = str(uuid.uuid4())    
    helpers.save_config(run_id, args)

    # Constants definition    
    zaxis_size = 642
    img_sz = [models.model_input_sizes[args.model_type], zaxis_size]
            
    # For reproducibility
    helpers.set_seed(args.seed)
        
    # Use the label map to assign labels to file paths
    val_file_paths = glob.glob(args.nii_path)
    
    # Create a list of zeros with the same length as val_file_paths
    val_file_labels = [0] * len(val_file_paths)

    output_dir = f"pred/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.load_model_and_state(
        args.model_type, 
        args.num_input_channels, 
        args.model_file, 
        device
    )
    
    val_dataloader_x, val_dataloader_y = datasets.create_eval_dataloaders(
        val_file_paths, 
        val_file_labels, 
        img_sz,
        args.central_prop_val, 
        args.planes_index_selection, 
        args.num_input_channels,
        args.batch_size,
        args.num_workers       
    )

    helpers.evaluate_and_save_predictions(
        model, 
        args.model_file, 
        val_dataloader_x, 
        val_dataloader_y, 
        device, 
        output_dir, 
        None, 
        args.nr_val_eval
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matterhorn Team - Workstream-2 Image Collaboratorium 2024 Training Script")
    parser.add_argument("--central-prop-val", type=float, default=0.40, help="Central proportion for validation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--nr-val-eval", type=int, default=3, help="Number of validation evaluations per axis")
    parser.add_argument("--model-type", choices=["B1", "B2", "B3", "B4", "B5", "B6", "B7"], default="B3", help="Model type")
    parser.add_argument("--nii-path", type=str, default="", help="Path to NII files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-mode", choices=["full", "half"], default="half", help="Training mode: full or half precision")
    parser.add_argument("--planes-index-selection", choices=["random", "consecutive"], default="random", help="")
    parser.add_argument("--num-input-channels", type=int, default=9, help="number of network input planes")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers of the DataLoader")
    parser.add_argument("--model-file", type=str, default="", help="Path to model file to evaluate")

    args = parser.parse_args()
    main(args)
