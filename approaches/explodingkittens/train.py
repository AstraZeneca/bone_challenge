from approaches.explodingkittens.src import get_metadata
from approaches.explodingkittens.src.preprocess import BoneProcessor

import argparse
import numpy as np
import lightning.pytorch as pl
from albumentations.pytorch import ToTensorV2
import albumentations as A
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from approaches.explodingkittens.src import BoneDataModule
from approaches.explodingkittens.src import BoneNet


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process metadata and images to generate output in a specified directory."
    )

    # Add required positional arguments
    parser.add_argument(
        "metadata_path",
        type=str,
        help="Path to the metadata file (see README).",
    )
    parser.add_argument(
        "images_path",
        type=str,
        help="Path to the directory containing images.",
    )

    # Add optional argument with a default value
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./2_5D_images",
        help="Path to the directory where output will be saved (default: ./2_5D_images).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create metadata df
    metadata_df = get_metadata(args.metadata_path, args.images_path)
    metadata_df.to_csv("./metadata.csv")

    # Process 3D bone scans to 2.5D
    print("Processing data...")
    processor = BoneProcessor(
        metadata_df, output_dir=args.output_dir, bit_depth=8, window=(2000, 500)
    )
    processor.process_files()

    # Define transforms
    train_transforms = A.Compose(
        [
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose([A.Normalize(), ToTensorV2()])

    # Cross-validation
    print("Training models...")
    _ = train_and_evaluate_kfold(
        args.output_dir, train_transforms, val_transforms, num_folds=4
    )


def train_and_evaluate_kfold(
    image_dir: str,
    train_transforms: A.Compose,
    val_transforms: A.Compose,
    num_folds: int,
    model_name: str = "efficientnet_es_pruned",
    batch_size: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-2,
    max_epochs: int = 12,
    log_dir: str = "logs",
    save_model: bool = True,
) -> float:

    scores = []

    for i in range(num_folds):
        dm = BoneDataModule(
            image_dir,
            batch_size=batch_size,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            splits=num_folds,
            fold=i,
        )
        model = BoneNet(model_name=model_name, lr=lr, wd=wd)

        logger = CSVLogger(log_dir, name=f"bonenet_fold{i}")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=[lr_monitor],
        )
        trainer.fit(model, dm)

        if save_model:
            trainer.save_checkpoint(f"bonenet_fold{i}.ckpt")

        scores.append(model.final_score)

    mean_score = np.mean(scores)
    print(f"Mean Score across {num_folds} folds: {mean_score}")

    return mean_score


if __name__ == "__main__":
    main()
