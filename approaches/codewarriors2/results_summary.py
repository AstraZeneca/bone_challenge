# importing libraries
import pandas as pd
import numpy as np
import cv2
from scipy.stats import norm
from datetime import datetime
import os

# path to file with predictions saved
PRED_PATH = './testing/predictions/model_20240510_112514_8_predictions.csv'
# path to file with training metadata
TRAIN_CSV = 'train.csv'
KERNEL_SIZE = 3
RESULTS_FOR_TESTING = True
RESULTS_OFFICIAL = True
RESULTS_FOLDER_PATH = './results/results_summary'


def _calculate_score(pred_slice_num, gt_slice_num):
    """Returns the survival function a single-sided normal distribution with stddev=3."""
    diff = abs(pred_slice_num - gt_slice_num)
    return 2 * norm.sf(diff, 0, 3)

def opening(prediction: list, kernel_size: int = 3) -> np.array:
    arr = np.asarray(prediction)
    kernel = np.ones((kernel_size, 1), np.uint8)
    cleaned_arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)

    return cleaned_arr


def get_full_result_table(pred_path: str = PRED_PATH,
                          train_csv: str = TRAIN_CSV,
                          kernel_size: int = KERNEL_SIZE,
                          results_for_testing: bool = RESULTS_FOR_TESTING,
                          results_official: bool = RESULTS_OFFICIAL) -> list[pd.DataFrame]:
    preds = pd.read_csv(pred_path)
    train = pd.read_csv(train_csv)

    preds[['Image Name', 'slice_idx']] = preds['img_file_name'].str.split('_', expand=True)
    preds['slice_idx'] = preds['slice_idx'].apply(lambda x: int(x.split('.')[0]))
    preds_gb = preds[['Image Name', 'prediction']].groupby('Image Name', as_index=False).agg(list)
    preds_gb['cleaned'] = preds_gb['prediction'].apply(lambda x: opening(x, kernel_size))
    preds_gb['first_after'] = preds_gb['cleaned'].apply(lambda x: np.where(x == 0)[0][0])

    # getting requested results
    results_off = None
    results = None
    if results_official:
        results_off = preds_gb[['Image Name', 'first_after']].rename(columns={'first_after': 'Growth Plate Index'})
    if results_for_testing:
        results = pd.merge(train, preds_gb, on='Image Name', how='right')
        results['score'] = results.apply(lambda row: _calculate_score(row['first_after'], row['Growth Plate Index']),
                                         axis=1)
        print("------ MODEL RESULTS ------")
        print(f"Mean model result: {np.mean(results['score'])}")
        print(f"Std model result: {np.std(results['score'])}")
        print(f"Score sum: {np.sum(results['score'])} [best: {preds_gb.shape[0]}]")

    return [results, results_off]


[results, results_off] = get_full_result_table()
if results is not None:
    results_file = os.path.join(RESULTS_FOLDER_PATH,
                                f'testing-results-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv')
    results.to_csv(results_file, index=False)

if results_off is not None:
    results_file = os.path.join(RESULTS_FOLDER_PATH,
                                f'official-results-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv')
    results_off.to_csv(results_file, index=False)
    