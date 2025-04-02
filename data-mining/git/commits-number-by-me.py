from github import Github
from github import Auth

if __name__ == '__main__':
    auth = Auth.Token()

    g = Github(auth=auth)

    for repo in g.get_user().get_repos():
        comment_numbers = {}
        comment_numbers[repo] = repo.get_commits().totalCount
        print(comment_numbers)
        
    g.close()
