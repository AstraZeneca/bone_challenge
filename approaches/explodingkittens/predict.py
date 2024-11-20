import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

from approaches.explodingkittens.src import get_all_images
from approaches.explodingkittens.src import BoneNet
from approaches.explodingkittens.src import BoneDatasetTest, ensemble_predict


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Do bone plate plane prediction on new data."
    )

    # Add required positional arguments
    parser.add_argument(
        "test_images_path",
        type=str,
        help="Path to the directory containing images.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Get test images and create dataloader
    print("Loading images...")
    img_paths = get_all_images(args.test_images_path, ext="nii")
    print(img_paths)

    test_transforms = A.Compose([A.Normalize(), ToTensorV2()])
    dst = BoneDatasetTest(img_paths, transforms=test_transforms)
    test_dataloader = DataLoader(
        dst, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load models
    print("Loading models...")
    models = load_models(n_folds=4)

    # Predict
    print("Predicting...")
    all_preds = []
    for _, images in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        avg_preds = ensemble_predict(images, models)
        print(avg_preds)
        all_preds.append(avg_preds)

    all_preds = np.concatenate(all_preds, axis=0)

    # Save predictions
    df = pd.DataFrame(
        {
            "Image Name": [Path(path).stem for path in img_paths],
            "Growth Plate Index": all_preds.squeeze().astype(int),
        }
    )

    df.sort_values(by="Image Name", inplace=True)
    df.to_csv("predictions.csv", index=False)


def load_models(n_folds: int, model_name: str = "bonenet"):
    models = [
        BoneNet.load_from_checkpoint(f"{model_name}_fold{fold}.ckpt")
        for fold in range(n_folds)
    ]

    return models


if __name__ == "__main__":
    main()
