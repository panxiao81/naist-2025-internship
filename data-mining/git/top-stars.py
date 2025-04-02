from github import Github
import html

if __name__ == '__main__':
    g = Github()
    print("Star Top 10")
    repos  = g.search_repositories(query="stars:>10",sort='stars', order='desc')
    for repo in repos[:10]:
        print(repo)

    print("Java star top 10")
    repos  = g.search_repositories(query="stars:>10 language:java",sort='stars', order='desc')
    for repo in repos[:10]:
        print(repo)

    print("python star top 10")
    repos  = g.search_repositories(query="stars:>10 language:python",sort='stars', order='desc')
    for repo in repos[:10]:
        print(repo)
