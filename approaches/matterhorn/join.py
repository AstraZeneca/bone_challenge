import os
import pandas as pd
import argparse

def process_csv_files(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dataframes.append(df)
    return dataframes

def calculate_and_save_result(dataframes, output_file, stat_type):
    result_df = pd.concat(dataframes)
    result_colname = result_df.columns[1]
    result_df = result_df.groupby('Image_Name').agg({result_colname: ['median', 'mean', 'std', 'count']})
    result_df.columns = ['median', 'mean', 'stdev', 'count']
    result_df['median'] = result_df['median'].round().astype("int")
    result_df['mean'] = result_df['mean'].round().astype("int")
    result_df = result_df.reset_index()
    result_df.to_csv("stats.csv", index=False)
    result_df.loc[:, ["Image_Name", stat_type]].to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing CSV files')
    parser.add_argument('--output', type=str, default='results.csv', help='Output file name (default: results.csv)')
    parser.add_argument('--stat', type=str, choices=['median', 'mean'], default='median', help='Statistic to use (median or mean, default: median)')
    args = parser.parse_args()

    csv_dataframes = process_csv_files(args.directory)
    calculate_and_save_result(csv_dataframes, args.output, args.stat)
