import pandas as pd
import numpy as np

train = pd.read_csv('data/full_train.csv')
is_valid = np.random.choice([True, False], size=len(train), p=[0.2, 0.8])
train['is_valid'] = is_valid
valid = train[train['is_valid'] == True]
train = train[train['is_valid'] == False]
valid.to_csv('data/valid.csv', index=False)
train.to_csv('data/train.csv', index=False)