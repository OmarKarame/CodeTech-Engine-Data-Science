import pandas as pd
from unidecode import unidecode
import re
import sys
import os

def python_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Takes Pandas DataFrame as input and returns o"""
    # Convert the 'diff' column to strings and handle NaN values
    df['diff'] = df['diff'].astype(str)

    return df[df['diff'].str.contains(r'\.py\b')].reset_index(drop=True)


def exclude_merge_commits(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude potential merge commits based on a heuristic."""
    df = df[~df['diff'].str.contains(r'Merge branch')]
    return df.reset_index(drop=True)

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

def large_diff_remover(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["diff_size"] = df_copy.apply(lambda x: sys.getsizeof(x["diff"]),axis=1)
    df_clean = df_copy.query("diff_size < 3000").reset_index(drop=True).drop("diff_size", axis=1)
    return df_clean

def pad_newlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spaces before and after newline characters in the 'diff' column to ensure tokenization doesn't join them with words.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A dataframe with newline characters in 'diff' column appropriately padded.
    """
    # Add spaces before and after newline characters
    df['diff'] = df['diff'].str.replace(r'(\S)(\n)(\S)', r'\1 \2 \3')

    return df.reset_index(drop=True)

def full_diff_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of preprocessing functions to clean and normalize the 'diff' column of a dataframe.

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'diff' column.

    Returns:
    - pd.DataFrame: A cleaned and normalized dataframe.
    """

    # Apply each function in sequence
    df_copy = df.copy()
    df_copy = large_diff_remover(df_copy)
    df_copy = python_filter(df_copy)
    df_copy = exclude_merge_commits(df_copy)
    df_copy = exclude_multi_file_diffs(df_copy)
    df_copy = pad_newlines(df_copy)

    return df_copy.reset_index(drop=True)


###########################################################################################################################################################################################
# Message Cleaner

#Do not run
def remove_emojis(text: str) -> str:
    """
    Remove emojis from a given text string.

    Args:
    - text (str): Input text string.

    Returns:
    - str: Text string with emojis removed.
    """
    # Define the pattern for emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def emoji_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes emojis from message column

    Args:
    - df (pd.DataFrame): Input dataframe containing the 'message' column.

    Returns:
    - pd.DataFrame: A dataframe with emoji characters in 'message' column removed.
    """
    # Add spaces before and after newline characters

    df_copy = df.copy()

    df_copy["message"] = df_copy["message"].apply(remove_emojis)
    return df_copy

python_dev_abbreviations = {
    "MNT": "Maintenance",
    "MAINT": "Maintenance",
    "WIP": "Work In Progress",
    "POC": "Proof Of Concept",
    "RFC": "Request For Comments",
    "PR": "Pull Request",
    "MR": "Merge Request",
    "CI": "Continuous Integration",
    "CD": "Continuous Deployment",
    "QA": "Quality Assurance",
    "LGTM": "Looks Good To Me",
    "RTM": "Ready To Merge",
    "PTAL": "Please Take A Look",
    "BR": "Bug Report",
    "CR": "Change Request",
    "DOCS": "Documentation",
    "TDD": "Test Driven Development",
    "BDD": "Behavior Driven Development",
    "OOP": "Object Oriented Programming",
    "FP": "Functional Programming",
    "UAT": "User Acceptance Testing",
    "API": "Application Programming Interface",
    "JWT": "JSON Web Token",
    "MVP": "Minimum Viable Product",
    "KISS": "Keep It Simple, Stupid",
    "DRY": "Don't Repeat Yourself",
    "YAGNI": "You Ain't Gonna Need It",
    "PEP": "Python Enhancement Proposal",
    "Ref": "Refactoring",
    "Init": "Initialization or Initial Commit",
    "GUI": "Graphical User Interface",
    "CLI": "Command Line Interface",
    "dep": "Dependencies",
    "cfg": "Configuration",
    "async": "Asynchronous",
    "await": "Asynchronous Wait",
    "IO": "Input/Output",
    "ex": "Exception",
    "func": "Function",
    "lib": "Library",
    "mod": "Module",
    "pkg": "Package",
    "gc": "Garbage Collector",
    "C-API": "Python C Application Programming Interface",
    "PR":"Pull Request",
    "BLD":"Build",
    "REL":"Release",
    "ENH":"Enhancement",
    "pdf":"Portable Document Format",
    "epub":"Electronic Publication",
    "csv":"Comma-Separated Values",
    "ASV":"Airspeed Velocity",
    "SVG":"Scalable Vector Graphics",
    "TST":"Test"
}
# DO not run
def replace_abbreviations(text):
    words = text.split()
    replaced_words = [python_dev_abbreviations.get(word.upper(), word) for word in words]
    return ' '.join(replaced_words)

def replace_abbreviations_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['message'] = df_copy['message'].apply(replace_abbreviations)
    return df_copy

def full_message_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = emoji_remover(df_copy)
    df_copy = replace_abbreviations_in_dataframe(df_copy)
    return df_copy



def process_files_in_directory(directory_path: str, output_file: str):
    # List to accumulate data from all JSONs
    all_data = []

    # Iterate over every file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            # Construct full path to the file
            full_path = os.path.join(directory_path, file_name)

            # Convert the JSON to a DataFrame
            df = pd.read_json(full_path)

            # Check for NaN values in the "diff" column and print the entire row

            # Apply the preprocessing function
            processed_df = full_diff_preprocessor(df)
            processed_df = full_message_preprocessor(processed_df)

            # Append data to our all_data list
            all_data.append(processed_df)

    # Concatenate all data into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_json(output_file)



if __name__ == "__main__":
    process_files_in_directory()
