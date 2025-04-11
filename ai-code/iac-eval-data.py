import datasets

dataset = datasets.load_dataset("autoiac-project/iac-eval")
print(dataset)

dataset['test'].to_csv("iac-eval-datasets.csv")