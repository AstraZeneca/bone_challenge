#  🏆 Matternhorn Approach 🏆

## 📋 Overview

This repository contains a modularized script that our team used to solve the Bone Challenge. Our approach involved using a deep learning model based on the EfficientNet family of models to predict the growth plane of a bone using NIfTI (nii) files. The project includes customizable parameters for various hyperparameters, allowing for different approaches to solving the problem.

## 🌟 Features
- **EfficientNet Models**: Utilizes the EfficientNet family for building the deep learning models.
- **Parametrization**: Allows for the customization of hyperparameters to explore various configurations.
- **Reproducible Research**: Designed with reproducible research in mind, tracking all runs in a `run_history.json` file. Each run is uniquely identified by a `run_id` and can only be executed on a committed git repository, ensuring that the script used and all hyperparameters are saved for each `run_id`.

## 🛠️ Installation

1. Create and activate a python environment.
2. Run `pip install -r requirements.txt`

## 💡 Usage
To train the model, run the script with the desired parameters. Here is an example command:
```bash
python train_model.py --param1 value1 --param2 value2
````

### ⚙️ Parameters

The following parameters can be used to customize the training process. Default are the ones that we used in our final submission.

- `--num-folds`: int (default: 5)
Number of folds for cross-validation.

- `--fold`: int (default: None)
If specified, run only one fold. Provide the fold number.

- `--epochs`: int (default: 2000)
Total number of epochs for training.

- `--central-prop`: float (default: 0.40)
Central proportion for training data.

- `--central-prop-val`: float (default: 0.40)
Central proportion for validation data.

- `--batch-size`: int (default: 16)
Size of the mini-batch used in training.

- `--nr-val-eval`: int (default: 10)
Number of validation evaluations per epoch.

- `--lr`: float (default: 3e-4)
Learning rate for the optimizer.

- `--model-type`: str (choices: "B1", "B2", "B3", "B4", "B5", "B6", "B7", default: "B3")
Type of the model to use for training.

- `--summary-path`: str 
Path to the CSV file where training summaries will be saved.

- `--nii-path`: str 
Path to the NII files used for training.

- `--seed`: int (default: 42)
Random seed for reproducibility.

- `--train-mode`: str (choices: "full", "half", default: "half")
Training precision mode: full or half precision.

- `--zaxis-mode`: str (choices: "crop", "interpolate", "mixed", default: "mixed")
Mode for z-axis handling: Crop, Interpolate, or Mixed.

- `--planes-index-selection`: str (choices: "random", "consecutive", default: "random")
Method for selecting planes index: Random or Consecutive.

- `--num-input-channels`: int (default: 9)
Number of input channels for the network.

- `--augmented-dataset-root-dir`: str 
Root directory for the augmented dataset.

- `--use-existing-augmented-dir`: bool (default: False)
Whether to use existing augmented datasets. If set, existing datasets will be used; otherwise, new ones will be generated.

- `--augmentation-train-count`: int (default: 150)
Number of augmentations per file for training.

- `--augmentation-valid-count`: int (default: 10)
Number of augmentations per file for validation.

- `--num-workers`: int (default: 4)
Number of workers for the DataLoader.


# Test reproducible

## Calculate the predictions

```bash

#!/bin/bash

# Define variables
MODEL_PATH=""
NII_PATH=""

MODEL_FILES=(
    "models/fold_0_epoch_1978_model_best_val_score_0.591_val_loss_10.1503.pt"
    "models/fold_1_epoch_1710_model_best_val_score_0.566_val_loss_11.6527.pt"
    "models/fold_2_epoch_806_model_best_val_score_0.602_val_loss_17.8721.pt"
    "models/fold_3_epoch_887_model_best_val_score_0.553_val_loss_23.7504.pt"
    "models/fold_4_epoch_1844_model_best_val_score_0.586_val_loss_16.3064.pt"
)

# Loop through each model file and run the prediction
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    python predict.py --model-file "$MODEL_FILE" --nii-path "$NII_PATH" --nr-val-eval 3
done

```

## Join the results

```
python join.py --output results/0.csv --stat median pred/2bed35d1-028e-463f-b0c3-44c6da60216d/
python join.py --output results/1.csv --stat median pred/357db53d-67ba-4672-9bad-934a478a682c/
python join.py --output results/2.csv --stat median pred/35a940e5-3344-4c14-a001-d5c4675717e0/
python join.py --output results/3.csv --stat median pred/f34d1167-3b2a-4b60-907f-429c7fbc391e/
python join.py --output results/4.csv --stat median pred/b69c6e0f-a68c-4d94-832f-30e9cf58a7c2/
python join.py --output results.csv --stat mean results/
cat y_true_test.csv | cut -d , -f 3,4 > y_true_test_only_preds.csv
python evaluate.py results.csv y_true_test_only_preds.csv 
```

## Test results

The output obtained from test set evaluation using B3-300 is:

Mean Score: 0.682
Standard Deviation of Score: 0.334
Sum of Scores: 8.87



| Arch | Size | Mean scr | Std scr | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------|------|----------|---------|--------|--------|--------|--------|--------|
| B0   | 300  |  0.578   |  0.017  |  0.605 |  0.577 |  0.580 |  0.559 |  0.570 |
| B1   | 300  |  0.604   |  0.053  |  0.667 |  0.616 |  0.585 |  0.626 |  0.526 | 
| B2   | 300  |  0.587   |  0.022  |  0.573 |  0.560 |  0.611 |  0.581 |  0.609 |
| B3   | 300  |  0.580   |  0.020  |  0.591 |  0.566 |  0.602 |  0.553 |  0.586 |
| B4   | 300  |  0.590   |  0.033  |  0.629 |  0.569 |  0.564 |  0.622 |  0.565 |
| B4   | 380  |  0.585   |  0.046  |  0.589 |  0.623 |  0.514 |  0.574 |  0.627 |
