import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from approaches.bytemeifyoucan.gpi_inferer import Config, GPIInferer, Predictor, Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=Path, nargs="+", required=True)
    parser.add_argument("--cache_rate", "-c", type=float, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    args = parser.parse_args()

    config = Config()

    predictor = Predictor(
        metadata=config.inp_config.metadata,
        images_path=config.inp_config.images_path,
        data_path=config.inp_config.data_path,
        trainer=Trainer(config),
        cache_rate=args.cache_rate,
    )

    for path in tqdm(args.model, desc="model"):
        data_path = path.with_suffix(".zarr")
        assert not data_path.exists()

        csv_path = path.with_suffix(".csv")
        assert not csv_path.exists()

        predictor.load_model(path)
        result = predictor.run(batch_size=args.batch_size, data_path=data_path)

        pd.DataFrame([
            GPIInferer(name=name, data=data).as_dict() for name, data in result.items()
        ]).to_csv(csv_path, index=False)
