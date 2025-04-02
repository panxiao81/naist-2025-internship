from git import Repo
import os

repo = Repo(os.path.join(os.environ.get("HOME"), "commons-math"))

assert not repo.bare

print(repo.head.commit.message)