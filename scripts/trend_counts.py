import pandas as pd
from pathlib import Path
ROOT=Path('c:/Users/ahadj/OneDrive/project')
gold=pd.read_csv(ROOT/'data'/'gold'/'model_dataset_monthly.csv',parse_dates=['month'],index_col='month')
trend_cols=[c for c in gold.columns if c.startswith('trends_') and '_lag' not in c]
print('trend cols:', trend_cols)
print('non-null counts:')
print(gold[trend_cols].notna().sum())
