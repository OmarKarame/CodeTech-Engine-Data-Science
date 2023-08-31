import pandas as pd
from unidecode import unidecode
import re

def python_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Takes Pandas DataFrame as input and returns o"""
    return df[df['diff'].str.contains(r'\.py\b')].reset_index(drop=True)

# def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalize whitespaces in the 'diff' column of the dataframe."""
#     df['diff'] = df['diff'].str.replace(r'\s+', ' ').str.strip()
#     return df.reset_index(drop=True)

def handle_non_ascii(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-ASCII characters to their closest ASCII equivalents."""
    df['diff'] = df['diff'].apply(unidecode)
    return df.reset_index(drop=True)

def normalize_file_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize file paths, masking user-specific details."""
    df['diff'] = df['diff'].str.replace(r'/home/[a-zA-Z0-9_]+/', '/home/user/')
    return df.reset_index(drop=True)

# def uniform_line_endings(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure uniform line endings by converting all to Unix-style."""
#     df['diff'] = df['diff'].str.replace(r'\r\n', '\n')
#     return df.reset_index(drop=True)

def exclude_merge_commits(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude potential merge commits based on a heuristic."""
    df = df[~df['diff'].str.contains(r'Merge branch')]
    return df.reset_index(drop=True)

# def filter_noise_commits(df: pd.DataFrame) -> pd.DataFrame:
#     """Exclude rows with noisy commit messages like 'WIP', 'fix', etc."""
#     noise_patterns = ['WIP', 'fix', 'minor change']
#     for pattern in noise_patterns:
#         df = df[~df['Git Commit Messages'].str.contains(pattern, case=False)]
#     return df.reset_index(drop=True)

def mask_dates_times(df: pd.DataFrame) -> pd.DataFrame:
    """Mask specific date and time references in the 'diff' column."""
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    df['diff'] = df['diff'].str.replace(date_pattern, '<DATE>').str.replace(time_pattern, '<TIME>')
    return df.reset_index(drop=True)

# def pad_newlines(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add spaces before and after newline characters in the 'diff' column to ensure tokenization doesn't join them with words.

#     Args:
#     - df (pd.DataFrame): Input dataframe containing the 'diff' column.

#     Returns:
#     - pd.DataFrame: A dataframe with newline characters in 'diff' column appropriately padded.
#     """
#     # Add spaces before and after newline characters
#     df['diff'] = df['diff'].str.replace(r'(\S)(\n)(\S)', r'\1 \2 \3')

#     return df.reset_index(drop=True)


# WARNING THE FUNCTION BELOW COULD LEAD TO LOTS OF DATA LOSS!!!
def exclude_multi_file_diffs(df: pd.DataFrame) -> pd.DataFrame:
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

import re

def replace_spaces_with_tabs(s):
    def replacer(match):
        num_spaces = len(match.group(0))
        num_tabs = num_spaces // 4
        leftover_spaces = num_spaces % 4
        return ' \t' * num_tabs + ' '

    # Directly replacing all sequences of spaces using the replacer function
    return re.sub(r' +', replacer, s)

def extract_changes_and_context(diff_str: str) -> tuple:
    """
    Extracts changes and contextual lines from the given diff string.

    Args:
    - diff_str (str): The diff string to process.

    Returns:
    - tuple: A tuple of strings, the first for changes and the second for context.
    """
    # Split the diff string into lines
    lines = diff_str.splitlines()

    # Initialize list to append clean lines to
    clean_diff = []

    # Initialize a flag to indicate if we're inside a hunk block
    in_hunk = False

    # Iterate through each line in the diff

    for i, line in enumerate(lines):
        # Skip the meta lines
        if line.startswith(("diff --git", "---", "+++", "index")):
            continue

        # Check for hunk header
        if line.startswith("@@"):
            in_hunk = True
            continue


        # If inside a hunk block
        if in_hunk:
            line = replace_spaces_with_tabs(line)
            if line.startswith("+"):
                clean_diff.append("[sad]"+line[1:]+"[eadd]")
            elif line.startswith("-"):
                clean_diff.append("[ssb]"+line[1:]+"[esb]")
            else:
                clean_diff.append("[scn]"+line+"[ecn]")

    # Convert lists to strings with added labels

    full_str = "\n".join(clean_diff)

    return full_str

def diff_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["diff"] = df_copy["diff"].apply(extract_changes_and_context)
    return df_copy


def full_preproc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of preprocessing functions to clean and normalize the 'diff' column of a dataframe.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A cleaned and normalized dataframe.
    """

    # Apply each function in sequence
    df = python_filter(df)
    
    df = normalize_whitespace(df)
    df = handle_non_ascii(df)
    df = normalize_file_paths(df)
    df = uniform_line_endings(df)
    df = exclude_merge_commits(df)
    df = mask_dates_times(df)
    df = pad_newlines(df)
    df = exclude_multi_file_diffs(df)

    return df.reset_index(drop=True)





if __name__ == "__main__":
    pass
