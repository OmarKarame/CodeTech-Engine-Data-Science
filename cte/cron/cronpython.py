from datetime import datetime
from cte.scraper.scraper import scrape_url, get_repo_commits, get_repo_diffs
import pandas as pd
import os

backend_directory = "/home/ben/code/OmarKarame/Commit-To-Excellence-Backend"

TOKEN = "temp"

HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3.diff'
}
variables = pd.read_csv(backend_directory + "/cte/cron/variables.csv")

# go through all repos provided in variables
for repo in variables[variables["not_finished"]]["repo"]:

    # go through all pages of a given repo
    while variables.query(f"repo == '{repo}'")["not_finished"][0]:
        page = int(variables.query(f"repo == '{repo}'")["current_page"])
        #print(repo,page)

        save_file = backend_directory + "/raw_data" + f"/{repo.split('/')[0]}_page_{page}.json"
        print(save_file)

        # check a page already exists
        if not os.path.exists(save_file):
            repo_commits = get_repo_commits(repo, HEADERS, page)

            repo_commits["diff"] = "error"
            if type(repo_commits) == int:
                raise Exception("could not get commit df")
            repo_commits.to_json(save_file,)

        else:
            repo_commits = pd.read_json(save_file)

        #check if there are repo_commits for a given page
        if len(repo_commits) == 0:
            variables["not_finished"] = variables.apply(lambda x : False if x["repo"] == repo else x["not_finished"], axis = 1)
            variables.to_csv(backend_directory + "/cte/cron/variables.csv", index = False, mode="w+")
            break

        #print(repo)
        repo_commits["diff"] = repo_commits.apply(lambda x : get_repo_diffs(repo, x["sha"], HEADERS) if x["diff"] == "error" else x["diff"], axis = 1)
        repo_commits.to_json(save_file)

        if len(repo_commits.query("diff == 'error'")) == 0:
            #if table is fully scraped update page
            variables["current_page"] = variables.apply(lambda x : page + 1 if x["repo"] == repo else x["current_page"], axis = 1)
            variables.to_csv(backend_directory + "/cte/cron/variables.csv", index = False, mode="w+")
        else:
            #if there are errors it means exceeded api reqest limit
            exit()
