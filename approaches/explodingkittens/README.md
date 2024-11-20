# Exploding Kittens Approach

## 2.5D BoneNet Training and Prediction

This repository provides scripts for training a model and predicting bone growth plate indices from 3D bone scans using a 2.5D approach.

## Installation

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Input Metadata File

The **training script** (`train.py`) requires a CSV metadata file with at least the following columns:

- **Image Name**: Image filenames without extensions.
- **Growth Plate Index**: The corresponding growth plate index.

### Example:

| Image Name | Growth Plate Index |
| ---------- | ------------------ |
| image1     | 1                  |
| image2     | 2                  |

## Training the Model (`train.py`)

To train the model, run:

```bash
python train.py <metadata_path> <images_path> --output_dir <output_directory>
```

- `<metadata_path>`: **Required** path to the metadata CSV file containing image names and growth plate indices.
- `<images_path>`: **Required** path to the directory with 3D images.
- `--output_dir`: Optional directory to save processed images (default: `./2_5D_images`).

Example:

```bash
python train.py metadata.csv /path/to/images --output_dir ./processed_images
```

### What it does:

1. Loads metadata and processes 3D bone scans into 2.5D images.
2. Trains the BoneNet model using k-fold cross-validation.
3. Saves logs and model checkpoints.

## Making Predictions (`predict.py`)

To predict growth plate indices for new images, run:

```bash
python predict.py <test_images_path>
```

- `<test_images_path>`: **Required** path to the directory with test images.

Example:

```bash
python predict.py /path/to/test_images
```

### What it does:

1. Loads test images and applies preprocessing.
2. Loads models from the k-fold training process.
3. Makes predictions by averaging outputs from all folds.
4. Saves predictions in a CSV file (`predictions.csv`).
