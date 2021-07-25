from os import read
import numpy as np
import pandas as pd

# 0 : Normal
# 1 : Offensive
# 2 : Hateful

def convert():
    data = pd.read_csv("datasets/fr_dataset.csv")
    readable = data[["tweet","sentiment"]].copy()
    print(readable["sentiment"].unique())
    for i,line in readable.iterrows():
        if "normal" in line["sentiment"]:
            readable.at[i,"sentiment"] = 0
        elif "offensive" in line["sentiment"]:
            readable.at[i,"sentiment"] = 1
        elif "hateful" in line["sentiment"]:
            readable.at[i,"sentiment"] = 2
        else:
            readable.at[i,"sentiment"] = 0
    print(readable["sentiment"].unique())
    readable.to_csv("datasets/fr_dataset_cleared.csv")

def loader():
    data = pd.read_csv("datasets/fr_dataset_cleared.csv", index_col = 0)
    msk = np.random.rand(len(data)) < 0.85
    train_loader = [(data.loc[i,"tweet"],data.loc[i,"sentiment"]) for i in range(len(msk)) if msk[i]]
    test_loader = [(data.loc[i,"tweet"],data.loc[i,"sentiment"]) for i in range(len(msk)) if not msk[i]]
    return train_loader,test_loader
