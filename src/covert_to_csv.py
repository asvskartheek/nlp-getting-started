# python covert_to_csv.py {model_arch}
import pandas as pd
import sys

model_name = sys.argv[1]

df_sample = pd.read_csv('data/sample_submission.csv')
df_results = pd.read_csv(f'experiments/predictions/{model_name}.csv',header=None)
df_results.columns = ['target']
df_sample['target'] = df_results['target']
df_sample.to_csv(f'experiments/predictions/{model_name}.csv', index=False)
