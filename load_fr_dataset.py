from os import read
import numpy as np
import pandas as pd
import random

# 0 : Normal
# 1 : Offensive
# 2 : Hateful

def convert():
    data = pd.read_csv("datasets/fr_dataset.csv")
    readable = data[["tweet","sentiment"]].copy()
    print(readable["sentiment"].unique())
    for i,line in readable.iterrows():
        if "normal" == line["sentiment"]:
            readable.at[i,"sentiment"] = 0
        elif "offensive" == line["sentiment"]:
            readable.at[i,"sentiment"] = 1
        elif "hateful" == line["sentiment"]:
            readable.at[i,"sentiment"] = 2
        else:
            readable.at[i,"sentiment"] = -1
            
    readable = readable[readable.sentiment != -1]
    readable.to_csv("datasets/fr_dataset_cleared_2.csv")

def equilibrate(load):
    counter = []
    l = [[],[],[]]
    for a,b in load:
        counter.append(b)
        l[b].append((a,b))
        
    c0 = counter.count(0)
    c1 = counter.count(1)
    c2 = counter.count(2)
    maxi = max(c0,c1,c2)
    while c0 < maxi/2:
        load.append(random.choice(l[0]))
        c0+=1
    while c1 < maxi/2:
        load.append(random.choice(l[1]))
        c1+=1
    while c2 < maxi/2:
        load.append(random.choice(l[2]))
        c2+=1

    counter = []
    for a,b in load:
        counter.append(b)

    random.shuffle(load)
    return load

def loader():
    data = pd.read_csv("datasets/fr_dataset_cleared_2.csv")
    msk = np.random.rand(len(data)) < 0.85
    train_loader = [(data.loc[i,"tweet"],data.loc[i,"sentiment"]) for i in range(len(msk)) if msk[i]]
    test_loader = [(data.loc[i,"tweet"],data.loc[i,"sentiment"]) for i in range(len(msk)) if not msk[i]]
    #equilibrate(train_loader)
    return train_loader, test_loader


if __name__ == "__main__":
    convert()