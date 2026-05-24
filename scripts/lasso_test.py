import pandas as pd
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
gold = pd.read_csv(ROOT / "data" / "gold" / "model_dataset_monthly.csv", parse_dates=["month"], index_col="month")
trend_cols = [c for c in gold.columns if c.startswith('trends_') and '_lag' not in c]
print('trend_cols=', trend_cols)
idx = gold.index.sort_values()
WINDOW=36
start_idx=0
window_idx = idx[start_idx:start_idx+WINDOW]
X = gold.loc[window_idx, trend_cols].copy()
y = gold.loc[window_idx, 'ipc_level'].copy()
df = pd.concat([y, X], axis=1)
print('window shape before dropna', df.shape)
print(df.head(10).to_string())
df = df.dropna()
print('window rows after dropna', df.shape[0])
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=5000))
try:
    pipeline.fit(df[trend_cols].values, df['ipc_level'].values)
    lasso = pipeline.named_steps['lassocv']
    print('alphas:', lasso.alphas_[:5])
    print('coef:', lasso.coef_)
except Exception as e:
    print('Lasso error', e)
