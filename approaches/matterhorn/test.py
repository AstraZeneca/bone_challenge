import argparse
import glob
import os
import uuid
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import helpers
import datasets
import models

def main(args):
    print("Using ensemble prediction with models trained by cross-validation.")
    
    run_id = str(uuid.uuid4())    
    helpers.save_config(run_id, args)

    zaxis_size = 642
    img_sz = [models.model_input_sizes[args.model_type], zaxis_size]    

    helpers.set_seed(args.seed)
        
    test_file_paths = glob.glob(args.nii_path)

    output_dir = f"pred/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_list = []
    for model_file in args.model_files:
        model = models.load_model_and_state(
            args.model_type, 
            args.num_input_channels, 
            model_file, 
            device
        )
        models_list.append(model)
    
    # Load test labels
    test_labels_df = pd.read_csv(args.test_labels_csv)
    
    test_dataloader_x, test_dataloader_y = datasets.create_eval_dataloaders(
        test_file_paths, 
        None, 
        img_sz, 
        args.central_prop_val, 
        args.planes_index_selection, 
        args.num_input_channels,
        args.batch_size,
        args.num_workers       
    )

    ensemble_predict_and_evaluate(
        models_list,
        test_dataloader_x, 
        test_dataloader_y, 
        device, 
        output_dir, 
        args.nr_eval_predictions,
        test_labels_df
    )

def ensemble_predict_and_evaluate(models, dataloader_x, dataloader_y, device, output_dir, nr_eval_predictions, test_labels_df):
    all_predictions = []
    all_image_names = []

    with torch.no_grad():
        for _ in range(nr_eval_predictions):
            batch_predictions = []
            for inputs, _ in dataloader_x:
                inputs = inputs.to(device)

                model_predictions = []
                for model in models:
                    model.eval()
                    outputs = model(inputs)
                    model_predictions.append(outputs.cpu().numpy())

                # Calculate mean of predictions across all models
                ensemble_prediction = np.mean(model_predictions, axis=0)
                batch_predictions.append(ensemble_prediction)

            all_predictions.append(batch_predictions)

        # Take the median of nr_eval_predictions
        median_predictions = np.median(all_predictions, axis=0)

        # Flatten the predictions and collect image names
        flattened_predictions = [pred for batch in median_predictions for pred in batch]
        all_image_names = [f.split('/')[-1].split('.')[0] for batch in dataloader_x.dataset for f in batch.image_paths]

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame({
        'Image Name': all_image_names,
        'Predicted Growth Plate Index': flattened_predictions
    })

    # Merge predictions with test labels
    merged_df = pd.merge(test_labels_df, predictions_df, on='Image Name')

    # Calculate evaluation metrics
    mse = mean_squared_error(merged_df['Growth Plate Index'], merged_df['Predicted Growth Plate Index'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(merged_df['Growth Plate Index'], merged_df['Predicted Growth Plate Index'])
    r2 = r2_score(merged_df['Growth Plate Index'], merged_df['Predicted Growth Plate Index'])

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")

    # Save predictions
    merged_df.to_csv(os.path.join(output_dir, 'predictions_with_labels.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matterhorn Team - Workstream-2 Image Collaboratorium 2024 Ensemble Prediction Script")
    parser.add_argument("--central-prop-val", type=float, default=0.40, help="Central proportion for validation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--nr-eval-predictions", type=int, default=10, help="Number of predictions per model for each input")
    parser.add_argument("--model-type", choices=["B1", "B2", "B3", "B4", "B5", "B6", "B7"], default="B3", help="Model type")
    parser.add_argument("--nii-path", type=str, default="", help="Path to NII files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--planes-index-selection", choices=["random", "consecutive"], default="random", help="")
    parser.add_argument("--num-input-channels", type=int, default=9, help="number of network input planes")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers of the DataLoader")
    parser.add_argument("--model-files", nargs='+', required=True, help="Paths to model files for ensemble prediction")
    parser.add_argument("--test-labels-csv", type=str, required=True, help="Path to CSV file containing test labels")

    args = parser.parse_args()
    main(args)