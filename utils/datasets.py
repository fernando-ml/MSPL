import os
import pandas as pd


def read_csv_files_in_folder(folder_path):
    """
    Reads CSV files from a folder and concatenates them into a single pandas DataFrame.

    Parameters:
    folder_path (str): Path to the folder containing the CSV files

    Returns:
    pandas.DataFrame: Concatenated DataFrame from all CSV files in the folder
    """
    # Get a list of all CSV files in the folder
    csv_files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Read each CSV file and append to a list of DataFrames
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    # Concatenate the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


