import os
import pandas as pd


def read_cicevse2024(folder_path):
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
        df = pd.read_csv(csv_file, low_memory=False)
        df['target'] = '-'.join(csv_file.removesuffix('.csv').split('-')
                                [3:]).lower()
        df['state'] = csv_file.removesuffix('.csv').split('-')[2]
        dfs.append(df)

    # Concatenate the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def read_cicov(folder_path):
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
        df = pd.read_csv(csv_file, low_memory=False)
        df['target'] = df['specific_class'].str.lower()
        dfs.append(df)

    # Concatenate the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


# Combine all CSV files in the CICEVSE2024
df = read_cicevse2024('data\CICEVSE2024')
# Save the combined DataFrame to a new Parquet file
df.to_parquet('cicevse_network.parquet', index=False)


# Combine all CSV files in the CICEVSE2024
df = read_cicov('data\CICIoV')
# Save the combined DataFrame to a new Parquet file
df.to_parquet('ciciov.parquet', index=False)
