import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

def evaluate_dataset(predicted_df, gt_df):
    merged_df = pd.merge(predicted_df, gt_df, left_on=predicted_df.columns[0], right_on=gt_df.columns[0], suffixes=('_pred', '_gt'), how='inner')
    scores = []
    for _, row in merged_df.iterrows():
        pred_slice_num = int(row.iloc[1])
        gt_slice_num = int(row.iloc[3]) 
        scores.append(_calculate_score(pred_slice_num, gt_slice_num))

    return np.mean(scores), np.std(scores), np.sum(scores)


def _calculate_score(pred_slice_num, gt_slice_num):
    diff = abs(pred_slice_num - gt_slice_num)
    return 2 * norm.sf(diff, 0, 3)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <predicted_file> <ground_truth_file>")
        sys.exit(1)

    predicted_file = sys.argv[1]
    ground_truth_file = sys.argv[2]

    predicted_df = pd.read_csv(predicted_file)
    gt_df = pd.read_csv(ground_truth_file)

    mean_score, std_score, sum_score = evaluate_dataset(predicted_df, gt_df)
    print("Mean Score:", mean_score)
    print("Standard Deviation of Score:", std_score)
    print("Sum of Scores:", sum_score)