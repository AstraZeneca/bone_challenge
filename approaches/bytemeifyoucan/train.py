import argparse
from pathlib import Path

from approaches.bytemeifyoucan.gpi_inferer import Config, Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=500)
    parser.add_argument("--fold", "-f", type=int, required=True)
    parser.add_argument("--save_interval", "-s", type=int, default=5)
    parser.add_argument("--cache_rate", "-c", type=float, default=1)
    parser.add_argument("--num_workers", "-w", type=int, default=1)
    parser.add_argument("--output_path", "-o", type=Path, required=True)
    parser.add_argument("--checkpoint", "-i", type=Path, default=None)
    args = parser.parse_args()

    config = Config()

    trainer = Trainer(
        config,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    trainer.run(
        fold=args.fold,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
    )
