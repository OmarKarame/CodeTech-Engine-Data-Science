import numpy as np
import pandas as pd
import requests

def scrape_url(url : str, headers : dict, params : dict = {}) -> dict | int:
    """scrape a url and return the json as a dict, if response is error returns it"""
    response = requests.get(url, headers = headers, params = params)
    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code

def get_repo_commits(repo : str, headers : dict, page : str = 1) -> list | int:
    """scrape all the commits from a repo for a given page number, if response is error returns it"""

    repo_url = f'https://api.github.com/repos/{repo}/commits'
    params = {
        "page" : page,
        'per_page':100
    }

    #returns a list of 30 commits
    response = scrape_url(repo_url, headers, params)
    print(type(response))

    #check that scrape returned a dict and not error code
    if type(response) == list:

        #get commit message from each commit in the response
        response_commit_messages = [commit["commit"]["message"] for commit in response]
        #get commit sha from each commit in the response
        response_commit_shas = [commit["sha"] for commit in response]

        return [response_commit_messages, response_commit_shas]

    else:
        #returns the response error
        return response

def get_repo_diffs(repo : str, sha : str, headers : dict):
    """scrape the git commit diff for a specified commit sha, if response is error returns it"""
    url = f'https://api.github.com/repos/{repo}/commits/{sha}'
    #response = scrape_url(url, headers)
    response = requests.get(url, headers=headers).text
    return response

def get_all_repo_commits(repo: str, headers : dict, tokens : list | str, max_page=10):
    '''scrape all commit messages and corresponding shas for a given repo'''
    if type(tokens) == str:
        tokens = [tokens]

    #make use of all provided keys for scraping
    for token in tokens:
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3.diff'
            }
        page = 1
        response = [["start"],["start"]]
