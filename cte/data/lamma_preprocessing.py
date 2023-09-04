import pandas as pd
from unidecode import unidecode
import re
import sys
import os

def lamma_python_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Takes Pandas DataFrame as input and returns o"""
    # Convert the 'diff' column to strings and handle NaN values
    df['diff'] = df['diff'].astype(str)

    return df[df['diff'].str.contains(r'\.py\b')].reset_index(drop=True)


def lamma_exclude_merge_commits(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude potential merge commits based on a heuristic."""
    df = df[~df['diff'].str.contains(r'Merge branch')]
    return df.reset_index(drop=True)

# WARNING THE FUNCTION BELOW COULD LEAD TO LOTS OF DATA LOSS!!!
def lamma_exclude_multi_file_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude rows from the dataframe where the 'diff' column suggests changes in multiple files.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A dataframe with rows corresponding to multi-file diffs excluded.
    """
    # Count occurrences of the pattern that indicates a file change
    file_change_count = df['diff'].str.count(r'diff --git')

    # Only keep rows where there's a single file change
    df = df[file_change_count == 1]

    return df.reset_index(drop=True)

# Do not run
def lamma_replace_spaces_with_tabs(s):
    def replacer(match):
        num_spaces = len(match.group(0))
        num_tabs = num_spaces // 4
        leftover_spaces = num_spaces % 4
        return ' \t' * num_tabs + ' '

    # Directly replacing all sequences of spaces using the replacer function
    return re.sub(r' +', replacer, s)


def lamma_large_diff_remover(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["diff_size"] = df_copy.apply(lambda x: sys.getsizeof(x["diff"]),axis=1)
    df_clean = df_copy.query("diff_size < 3000").reset_index(drop=True).drop("diff_size", axis=1)
    return df_clean


def full_diff_preprocessor_for_lamma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of preprocessing functions to clean and normalize the 'diff' column of a dataframe.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A cleaned and normalized dataframe.
    """

    # Apply each function in sequence
    df_copy = df.copy()
    df_copy = lamma_large_diff_remover(df_copy)
    df_copy = lamma_python_filter(df_copy)
    df_copy = lamma_exclude_merge_commits(df_copy)
    df_copy = lamma_exclude_multi_file_diffs(df_copy)
    return df_copy.reset_index(drop=True)

def lamma_process_files_in_directory(directory_path: str, output_file: str):
    # List to accumulate data from all JSONs
    all_data = []

    # Iterate over every file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            # Construct full path to the file
            full_path = os.path.join(directory_path, file_name)

            # Convert the JSON to a DataFrame
            df = pd.read_json(full_path).dropna()

            # Copy for posterity
            df_copy = df.copy()

            # Apply the preprocessing functions
            processed_df = full_diff_preprocessor_for_lamma(df_copy)

            # Append data to our all_data list
            all_data.append(processed_df)

    # Concatenate all data into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_json(output_file)
    return final_df

if __name__ == "__main__":
    pass
