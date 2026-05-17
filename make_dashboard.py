"""
make_dashboard.py — Figure de synthese finale BESI V3
Genere outputs/figures/dashboard_besi_v3_final.png
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
GOLD    = ROOT / 'data/gold/model_dataset_monthly.csv'
BT_PATH = ROOT / 'outputs/reports/backtest_v3_results.csv'
WM_PATH = ROOT / 'outputs/reports/warning_metrics_v3.csv'
FIG_DIR = ROOT / 'outputs/figures'

df = pd.read_csv(GOLD, parse_dates=['month'], index_col='month')
bt = pd.read_csv(BT_PATH)
wm = pd.read_csv(WM_PATH)

BREAK  = pd.Timestamp('2022-03-01')
C_IPC  = '#2c3e50'
C_PRE  = '#3498db'
C_POST = '#e74c3c'
C_BESI = '#8e44ad'
C_NAI  = '#95a5a6'
C_SAR  = '#3498db'
C_SAX  = '#e74c3c'

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25})

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#fafafa')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)

# ── Panel 1 (top, spans 2 cols) : IPC + Inflation + rupture ──────────────────
ax1  = fig.add_subplot(gs[0, :2])
ax1b = ax1.twinx()
ipc  = df['ipc_level'].dropna()

if 'inflation_yoy' in df.columns:
    yoy  = df['inflation_yoy'].dropna()
    pre  = yoy[yoy.index <  BREAK]
    post = yoy[yoy.index >= BREAK]
    ax1b.fill_between(pre.index,  0, pre.values,  alpha=0.18, color=C_PRE)
    ax1b.fill_between(post.index, 0, post.values, alpha=0.18, color=C_POST)
    ax1b.plot(pre.index,  pre.values,  color=C_PRE,  lw=1.4, label='YoY pre-2022')
    ax1b.plot(post.index, post.values, color=C_POST, lw=1.4, label='YoY post-2022')
    ax1b.axhline(2.0, color='orange', ls=':', lw=1, alpha=0.8)
    ax1b.set_ylabel('Inflation YoY (%)', color='gray', fontsize=8)
    ax1b.tick_params(axis='y', labelcolor='gray', labelsize=8)

ax1.plot(ipc.index, ipc.values, color=C_IPC, lw=2.2, label='IPC niveau', zorder=5)
ax1.axvline(BREAK, color='black', ls='--', lw=1.5, label='Rupture mars 2022')
ax1.set_ylabel('IPC (base 2017=100)', fontsize=9)
ax1.set_title('IPC Maroc & Inflation YoY — Rupture structurelle 2022\n'
              '(Delta moy: +7.8 pts | x11.6 | p<0.0001)', fontsize=10, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax1b.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, fontsize=7, loc='upper left')
ax1.tick_params(labelsize=8)

# ── Panel 2 (top-right) : scatter BESI vs Inflation ──────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
if 'behavioral_index_pure' in df.columns and 'inflation_yoy' in df.columns:
    besi = df['behavioral_index_pure'].dropna()
    yoy2 = df['inflation_yoy'].reindex(besi.index)
    mask = ~np.isnan(yoy2.values)
    sc   = ax2.scatter(besi.values[mask], yoy2.values[mask],
                       c=yoy2.values[mask], cmap='RdYlGn_r',
                       alpha=0.7, s=35, vmin=-2, vmax=10)
    z  = np.polyfit(besi.values[mask], yoy2.values[mask], 1)
    xf = np.linspace(besi.min(), besi.max(), 50)
    ax2.plot(xf, np.polyval(z, xf), 'k--', lw=1.3, alpha=0.65, label='r=+0.54*')
    ax2.axhline(2.0, color='orange', ls=':', lw=1.2, label='Seuil 2%')
    ax2.set_xlabel('BESI behavioral', fontsize=9)
    ax2.set_ylabel('Inflation YoY (%)', fontsize=9)
    ax2.set_title('BESI vs Inflation\n(r=0.535, p<0.001)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

# ── Panel 3 (mid-left) : Backtest RMSE par bloc ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
model_colors = {'naif': C_NAI, 'sarima': C_SAR, 'sarimax_behavioral': C_SAX}
model_labels  = {'naif': 'Naif', 'sarima': 'SARIMA', 'sarimax_behavioral': 'SARIMAX+BESI'}
blocs = sorted(bt['bloc'].unique())
x3    = np.arange(len(blocs))
w3    = 0.25
for i, (model, grp) in enumerate(bt.groupby('model')):
    vals  = grp.set_index('bloc')['rmse'].reindex(blocs).values
    bars3 = ax3.bar(x3 + i*w3, vals, w3,
                    label=model_labels.get(model, model),
                    color=model_colors.get(model, 'gray'), alpha=0.85)
    for bar, v in zip(bars3, vals):
        if not np.isnan(v):
            ax3.text(bar.get_x()+bar.get_width()/2, v+0.02,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
ax3.set_xticks(x3 + w3)
ax3.set_xticklabels([f'Bloc {b}' for b in blocs], fontsize=9)
ax3.set_title('RMSE Backtest walk-forward\n1-step-ahead', fontsize=10, fontweight='bold')
ax3.set_ylabel('RMSE (pts IPC)', fontsize=9)
ax3.legend(fontsize=7)
ax3.set_ylim(0, 2.6)
ax3.tick_params(labelsize=8)

# ── Panel 4 (mid-center) : AIC SARIMA vs SARIMAX ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
aic_vals   = [64.85, 57.09]
aic_labels = ['SARIMA', 'SARIMAX\n+BESI']
bars4 = ax4.bar(aic_labels, aic_vals, color=[C_SAR, C_SAX], alpha=0.85, width=0.5)
for bar, v in zip(bars4, aic_vals):
    ax4.text(bar.get_x()+bar.get_width()/2, v+0.3,
             f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.annotate('', xy=(1, 57.09), xytext=(0, 64.85),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax4.text(0.5, 61.0, 'AIC = -7.77', ha='center', fontsize=9, style='italic')
ax4.set_ylim(50, 73)
ax4.set_ylabel('AIC (Train A)', fontsize=9)
ax4.set_title('AIC In-Sample (Train A)\nFaveur SARIMAX+BESI', fontsize=10, fontweight='bold')
ax4.tick_params(labelsize=9)

# ── Panel 5 (mid-right) : Early warning par bloc ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
metric_cols  = ['auc', 'f1', 'precision', 'recall']
metric_labs  = ['AUC', 'F1', 'Precision', 'Recall']
w5 = 0.35
x5 = np.arange(len(metric_cols))
wm_a = wm[wm['scope'] == 'test_A']
wm_b = wm[wm['scope'] == 'test_B']
if not wm_a.empty:
    va = [float(wm_a[c].iloc[0]) for c in metric_cols]
    ax5.bar(x5 - w5/2, va, w5, label='Bloc A (COVID)', color=C_PRE, alpha=0.8)
if not wm_b.empty:
    vb = [float(wm_b[c].iloc[0]) for c in metric_cols]
    bars5 = ax5.bar(x5 + w5/2, vb, w5, label='Bloc B (Inflation)', color=C_POST, alpha=0.8)
    for xi, v in zip(x5 + w5/2, vb):
        ax5.text(xi, v+0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
ax5.axhline(0.65, color='orange', ls=':', lw=1.2, label='Seuil H1')
ax5.set_xticks(x5); ax5.set_xticklabels(metric_labs, fontsize=9)
ax5.set_ylim(0, 1.18)
ax5.set_title('Early Warning\nBloc B : Recall=1.00 (100% eps. detectes)', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7, loc='upper right')
ax5.tick_params(labelsize=8)

# ── Panel 6 (bottom, spans 3) : Predictions backtest timeline ────────────────
ax6 = fig.add_subplot(gs[2, :])
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
    ipc_full  = df['ipc_level'].dropna()
    exog_col  = 'behavioral_index_pure_lag1'
    bloc_defs = [
        ('A', 'train_A', 'test_A'),
        ('B', 'train_B', 'test_B'),
    ]
    all_dates, all_act, all_sar, all_sax = [], [], [], []

    for bloc, trlbl, telbl in bloc_defs:
        test_idx = df[df['split_label'].str.contains(telbl)].index
        for t_idx in test_idx:
            cutoff = ipc_full.index.get_loc(t_idx)
            hist   = ipc_full.iloc[:cutoff]
            if len(hist) < 16:
                continue
            actual = float(ipc_full.loc[t_idx])
            # SARIMA
            try:
                m  = SM(hist, order=(1,1,1), seasonal_order=(1,0,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
                f  = m.fit(disp=False)
                ps = float(f.forecast(1).iloc[0])
            except Exception:
                ps = np.nan
            # SARIMAX
            psx = np.nan
            if exog_col in df.columns:
                exog_hist = df[exog_col].reindex(hist.index).fillna(method='ffill')
                exog_fore = df.loc[t_idx, exog_col]
                if not pd.isna(exog_fore):
                    try:
                        mx  = SM(hist, exog=exog_hist, order=(1,1,1), seasonal_order=(1,0,1,12),
                                 enforce_stationarity=False, enforce_invertibility=False)
                        fx  = mx.fit(disp=False)
                        psx = float(fx.forecast(1, exog=[[exog_fore]]).iloc[0])
                    except Exception:
                        pass
            all_dates.append(t_idx)
            all_act.append(actual)
            all_sar.append(ps)
            all_sax.append(psx)

    ax6.plot(all_dates, all_act, 'o-', color=C_IPC, lw=2.2, ms=4, label='IPC reel', zorder=5)
    ax6.plot(all_dates, all_sar, '--',  color=C_SAR, lw=1.4, label='SARIMA', alpha=0.85)
    ax6.plot(all_dates, all_sax, '--',  color=C_SAX, lw=1.4, label='SARIMAX+BESI', alpha=0.85)
    ax6.axvline(BREAK, color='black', ls='--', lw=1.2, alpha=0.6)
    ax6.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2022-01-01'),
                alpha=0.06, color='blue', label='Test A')
    ax6.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'),
                alpha=0.06, color='red',  label='Test B')
    ax6.set_ylabel('IPC (base 2017=100)', fontsize=9)
    ax6.set_title('Predictions 1-step-ahead — Walk-forward backtest\n'
                  'Apprentissage expansif : chaque prediction entraine sur tout l\'historique disponible',
                  fontsize=10, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper left', ncol=5)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax6.get_xticklabels(), rotation=25, fontsize=8)
    ax6.tick_params(axis='y', labelsize=8)
except Exception as e:
    ax6.text(0.5, 0.5, f'Predictions non affichees: {e}',
             ha='center', va='center', transform=ax6.transAxes, fontsize=9)

# ── Titre et sauvegarde ──────────────────────────────────────────────────────
fig.suptitle('BESI MAROC V3 — Behavioral Economic Stress Index\n'
             'Rupture 2022 | Backtest SARIMA vs SARIMAX+BESI | Alerte precoce',
             fontsize=13, fontweight='bold', y=0.995)

out = FIG_DIR / 'dashboard_besi_v3_final.png'
plt.savefig(out, bbox_inches='tight', dpi=150, facecolor='#fafafa')
plt.close()
print(f'Dashboard sauvegarde : {out}  ({out.stat().st_size//1024} KB)')
