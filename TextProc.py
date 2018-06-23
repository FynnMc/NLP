import numpy as np
import pandas as pd
import pyprind
import os

basepath = 'aclImdb'

labels = {'pos': 1, 'neg' : 0}
pbar = pyprind.ProgBar(50000) # Allows us to track progress
df = pd.DataFrame() # Empty DF object

## Nested for loops to generate text array for csv conversion
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

np.random.seed(123)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
