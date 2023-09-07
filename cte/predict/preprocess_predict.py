import pandas as pd
from unidecode import unidecode
import re
import sys
import os

# Do not run
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
                clean_diff.append("[sad]"+line[1:]+"[ead]")
            elif line.startswith("-"):
                clean_diff.append("[ssb]"+line[1:]+"[esb]")
            else:
                clean_diff.append("[scn]"+line+"[ecn]")

    # Convert lists to strings with added labels

    full_str = "\n".join(clean_diff)

    return full_str

def pad_newlines(diff):
    """
    Add spaces before and after newline characters in the 'diff' column to ensure tokenization doesn't join them with words.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A dataframe with newline characters in 'diff' column appropriately padded.
    """
    # Add spaces before and after newline characters
    diff = diff.replace(r'(\S)(\n)(\S)', r'\1 \2 \3')

    return diff

def full_diff_preprocessor(diff):
    """
    Apply a series of preprocessing functions to clean and normalize the 'diff' column of a dataframe.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A cleaned and normalized dataframe.
    """

    # Apply each function in sequence
    diff = replace_spaces_with_tabs(diff)
    diff = extract_changes_and_context(diff)
    diff = pad_newlines(diff)

    return diff
