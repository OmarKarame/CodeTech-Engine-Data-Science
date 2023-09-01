import click
from git import Repo
import git
import os

def find_git_directory():
    path = os.getcwd()
    while path != "/":
        if os.path.exists(os.path.join(path, '.git')):
            return path
        path = os.path.dirname(path)
    return None

def commit_empty_message():
    repo_path = find_git_directory()
    repo = git.Repo(f'{repo_path}')
    index = repo.index
    for item in repo.index.diff(None):
        item.commit("")
        click.echo(f'File {item.a_path} committed!')






# https://github.com/OmarKarame/Commit-To-Excellence-Backend
# OmarKarame/Commit-To-Excellence-Backend
# /Users/omarkarame/code/OmarKarame/Commit-To-Excellence/Commit-To-Excellence-Backend
# Code-Test-click-application-with-python-methods#34


def connect_py():
    """Simple program that connects to repo to execute git commands"""
    repo_path = find_git_directory()
    repo = git.Repo(f'{repo_path}')
    # click.echo(f"Connected SUCCESSFULLY to: {repo}!")
    return repo


@click.command()
def connect():
    """Simple program that connects to repo to execute git commands"""
    repo_path = find_git_directory()
    repo = git.Repo(f'{repo_path}')
    click.echo(f"Connected SUCCESSFULLY to: {repo}!")
    return repo


@click.command()
@click.option('-b','--branch', prompt='Checkout to branch',
              help='The branch you would like to change to/ work on')
def checkout(branch):
    """Simple program that allows user to checkout to another branch"""
    repo = connect_py()
    repo.git.checkout(branch)


@click.command()
def status():
    """Simple program that prints git status"""
    repo = connect_py()
    click.echo(repo.git.status())


@click.command()
@click.option('-f','--file', prompt='Add file',
              help='The file you want to add to your commit')
def add_file(file):
    """Simple program that allows the user to add a file to commit"""
    repo = connect_py()
    repo.git.add(os.path.abspath(file))


@click.command()
def get_diff():
    """Simple program that allows the user to add a file to commit"""
    repo = connect_py()
    index = repo.index
    # Get staged (added) files
    staged_files = [item.a_path for item in repo.index.diff(None)]

    diffs = repo.index.diff(None)

    for diff in diffs:
        print("File:", diff.a_path)
        print("Change Type:", diff.change_type)

        if diff.a_blob:  # Check if a_blob is not None
            print("Old Content:", diff.a_blob.data_stream.read())

        if diff.b_blob:  # Check if b_blob is not None
            print("New Content:", diff.b_blob.data_stream.read())

        print("Diff:\n", diff.diff)
        print("-" * 40)


if __name__ == '__main__':
    commit_empty_message()
