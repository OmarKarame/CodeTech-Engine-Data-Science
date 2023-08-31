import requests
import pandas as pd
import time

# Constants
API_KEYS = ["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", "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", "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"]  # add more as needed  # add more as needed

HEADERS = {
    'Accept': 'application/vnd.github.v3.diff'
}
PARAMS = {"per_page":"100"}
BASE_URL = 'https://api.github.com/repos'

def set_token(token):
    """Update the Authorization header with the provided token."""
    HEADERS['Authorization'] = f'token {token}'

def get_reset_time(response):
    """Get the reset time from the API response headers."""
    reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
    return reset_timestamp - int(time.time())  # returns remaining seconds

def get_all_commits(repo, start_url):
    url = start_url
    all_commits = []

    while url:
        response = requests.get(url, headers=HEADERS, params = PARAMS)
        if response.status_code == 200:
            all_commits.extend(response.json())
            link_header = response.headers.get('Link', '')
            if 'rel="next"' in link_header:
                url = [link.split(';')[0].strip('<>') for link in link_header.split(',') if 'rel="next"' in link][0]
            else:
                url = None
        elif response.status_code == 403:
            sleep_time = get_reset_time(response)
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time + 10)  # Adding an extra 10 seconds as a buffer
            continue
        else:
            # Enhanced error message
            print(f"Failed to retrieve commits for {repo}. Status Code: {response.status_code}. Message: {response.text}")
            url = None

    return all_commits, None



repos = ['scipy/scipy']
data = []

for repo in repos:
    next_url = f"{BASE_URL}/{repo}/commits"  # Initial URL

    while next_url:  # This loop will keep running until all commits have been scraped
        for token in API_KEYS:
            set_token(token)  # set the current token

            commits, next_url = get_all_commits(repo, next_url)
            for commit in commits:
                sha = commit['sha']
                message = commit['commit']['message']
                diff_response = requests.get(f"{BASE_URL}/{repo}/commits/{sha}", headers=HEADERS, params = PARAMS)

                if diff_response.status_code == 200:
                    diff = diff_response.text
                elif diff_response.status_code == 403:
                    sleep_time = get_reset_time(diff_response)
                    print(f"Rate limit reached when fetching diff. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time + 10)
                    continue
                else:
                    print(f"Failed to retrieve diff for commit {sha} in {repo}")
                    diff = ''

                data.append({
                    'Repository': repo,
                    'Message': message,
                    'Diff': str(diff)
                })

            if not next_url:  # If no more commits to scrape, break out of the token loop
                break

df.to_csv("data6")
