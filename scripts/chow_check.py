import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
gold = pd.read_csv(ROOT / 'data' / 'gold' / 'model_dataset_monthly.csv', parse_dates=['month'], index_col='month')
trend_cols = [c for c in gold.columns if c.startswith('trends_') and '_lag' not in c]
break_date = pd.Timestamp('2022-03-01')
cols_needed = ['ipc_level'] + trend_cols
pre = gold.loc[gold.index < break_date, cols_needed].dropna()
post = gold.loc[gold.index >= break_date, cols_needed].dropna()
print('trend_cols', trend_cols)
print('pre rows', pre.shape[0])
print('post rows', post.shape[0])
print('pre head:\n', pre.head().to_string())
print('post head:\n', post.head().to_string())
