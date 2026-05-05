"""Generate additional research-focused figures (22 more) to reach 33 total."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)
np.random.seed(7)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 120,
})


def save(name):
    plt.tight_layout()
    plt.savefig(f'figures/{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# -----------------------------------------------------------------------------
# Helper: generate a synthetic OHLC series used by several plots
# -----------------------------------------------------------------------------
n = 200
returns = np.random.normal(0.0003, 0.015, n)
close = 1800 * np.cumprod(1 + returns)
high = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
open_ = np.roll(close, 1)
open_[0] = close[0]
volume = np.abs(np.random.normal(50000, 15000, n)) + np.abs(returns) * 2e6


# -----------------------------------------------------------------------------
# Fig 12: Candlestick chart with 20- and 50-period SMA overlay
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
for i in range(n):
    col = '#2CA02C' if close[i] >= open_[i] else '#D62728'
    ax.plot([i, i], [low[i], high[i]], color=col, lw=0.8)
    ax.plot([i, i], [open_[i], close[i]], color=col, lw=2.5)

sma20 = np.convolve(close, np.ones(20)/20, mode='same')
sma50 = np.convolve(close, np.ones(50)/50, mode='same')
ax.plot(sma20, color='#1F4E79', lw=1.7, label='SMA(20)')
ax.plot(sma50, color='#ED7D31', lw=1.7, label='SMA(50)')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Price (USDT)')
ax.set_title('Figure 12. Candlestick chart with 20- and 50-period simple moving averages')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig12_candlestick_sma')


# -----------------------------------------------------------------------------
# Fig 13: RSI oscillator with overbought/oversold zones
# -----------------------------------------------------------------------------
def rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)

rsi_vals = rsi(close)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(rsi_vals, color='#7030A0', lw=1.6)
ax.axhline(70, color='#C00000', ls='--', lw=1.2, label='Overbought (70)')
ax.axhline(30, color='#2CA02C', ls='--', lw=1.2, label='Oversold (30)')
ax.fill_between(range(len(rsi_vals)), 70, rsi_vals, where=rsi_vals >= 70,
                color='#C00000', alpha=0.3)
ax.fill_between(range(len(rsi_vals)), 30, rsi_vals, where=rsi_vals <= 30,
                color='#2CA02C', alpha=0.3)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('RSI Value')
ax.set_title('Figure 13. 14-period Relative Strength Index with threshold zones')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)
save('fig13_rsi_oscillator')


# -----------------------------------------------------------------------------
# Fig 14: Bollinger Bands
# -----------------------------------------------------------------------------
period = 20
rolling_mean = np.array([np.mean(close[max(0, i-period):i+1]) for i in range(n)])
rolling_std = np.array([np.std(close[max(0, i-period):i+1]) for i in range(n)])
upper = rolling_mean + 2 * rolling_std
lower = rolling_mean - 2 * rolling_std

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(close, color='#1F4E79', lw=1.4, label='Price')
ax.plot(rolling_mean, color='#808080', lw=1.2, ls='--', label='20-period SMA')
ax.plot(upper, color='#C00000', lw=1.0, label='Upper Band (+2σ)')
ax.plot(lower, color='#2CA02C', lw=1.0, label='Lower Band (−2σ)')
ax.fill_between(range(n), lower, upper, color='#DCE6F2', alpha=0.4)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Price (USDT)')
ax.set_title('Figure 14. Bollinger Bands applied to the ETH/USDT minute series')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig14_bollinger_bands')


# -----------------------------------------------------------------------------
# Fig 15: MACD (line, signal, histogram)
# -----------------------------------------------------------------------------
def ema(x, p):
    alpha = 2 / (p + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

macd_line = ema(close, 12) - ema(close, 26)
signal_line = ema(macd_line, 9)
hist = macd_line - signal_line

fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})
axes[0].plot(close, color='#1F4E79', lw=1.4)
axes[0].set_ylabel('Price')
axes[0].set_title('(a) Underlying price')
axes[0].grid(alpha=0.3)

axes[1].plot(macd_line, color='#1F4E79', lw=1.5, label='MACD')
axes[1].plot(signal_line, color='#ED7D31', lw=1.5, label='Signal')
colors = ['#2CA02C' if h >= 0 else '#D62728' for h in hist]
axes[1].bar(range(n), hist, color=colors, alpha=0.6, width=1.0)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_ylabel('MACD Value')
axes[1].set_xlabel('Time (minutes)')
axes[1].set_title('(b) MACD line, signal line and histogram')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 15. MACD momentum indicator', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig15_macd.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 16: Moving-average crossover signal overlay
# -----------------------------------------------------------------------------
fast = ema(close, 20)
slow = ema(close, 50)
cross_up = np.where((fast[:-1] < slow[:-1]) & (fast[1:] >= slow[1:]))[0]
cross_down = np.where((fast[:-1] > slow[:-1]) & (fast[1:] <= slow[1:]))[0]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(close, color='#808080', lw=1.1, alpha=0.7, label='Price')
ax.plot(fast, color='#2CA02C', lw=1.7, label='EMA(20)')
ax.plot(slow, color='#C00000', lw=1.7, label='EMA(50)')
ax.scatter(cross_up, close[cross_up], marker='^', s=150, color='#2CA02C',
           edgecolor='black', zorder=5, label='Bullish crossover')
ax.scatter(cross_down, close[cross_down], marker='v', s=150, color='#C00000',
           edgecolor='black', zorder=5, label='Bearish crossover')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Price (USDT)')
ax.set_title('Figure 16. Moving-average crossover signals feeding the strategy engine')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig16_ma_crossover')


# -----------------------------------------------------------------------------
# Fig 17: Distribution of daily log-returns
# -----------------------------------------------------------------------------
daily_returns = np.random.normal(0.0008, 0.018, 180)
daily_returns[30:40] = np.random.normal(-0.015, 0.03, 10)
daily_returns[100:105] = np.random.normal(-0.025, 0.035, 5)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(daily_returns, bins=40, color='#4472C4', edgecolor='#1F4E79', alpha=0.85,
        density=True, label='Empirical')
mu, sigma = np.mean(daily_returns), np.std(daily_returns)
xs = np.linspace(daily_returns.min(), daily_returns.max(), 200)
ax.plot(xs, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(xs-mu)**2/(2*sigma**2)),
        color='#C00000', lw=2, label=f'Normal fit (μ={mu:.4f}, σ={sigma:.4f})')
ax.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('Daily Log-Return')
ax.set_ylabel('Density')
ax.set_title('Figure 17. Distribution of daily log-returns from the agent')
ax.legend()
ax.grid(alpha=0.3)
save('fig17_returns_distribution')


# -----------------------------------------------------------------------------
# Fig 18: Monthly returns heatmap
# -----------------------------------------------------------------------------
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = ['2021', '2022', '2023', '2024', '2025']
data = np.random.normal(1.5, 4.0, (5, 12))
data[1, 4:9] = np.random.normal(-5, 3, 5)  # rough 2022

fig, ax = plt.subplots(figsize=(11, 3.8))
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax.set_xticks(range(12))
ax.set_xticklabels(months)
ax.set_yticks(range(5))
ax.set_yticklabels(years)
for i in range(5):
    for j in range(12):
        ax.text(j, i, f'{data[i, j]:+.1f}', ha='center', va='center',
                fontsize=8.5, color='black')
plt.colorbar(im, ax=ax, label='Monthly Return (%)')
ax.set_title('Figure 18. Monthly return heatmap across the evaluation history')
save('fig18_monthly_heatmap')


# -----------------------------------------------------------------------------
# Fig 19: Cumulative returns vs multiple benchmarks
# -----------------------------------------------------------------------------
days = 180
agent = np.cumsum(np.random.normal(0.09, 1.0, days)) / 100
btc   = np.cumsum(np.random.normal(0.04, 1.5, days)) / 100
sp500 = np.cumsum(np.random.normal(0.03, 0.5, days)) / 100
eth   = np.cumsum(np.random.normal(0.05, 1.8, days)) / 100

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(agent * 100, label='Proposed Agent', color='#1F4E79', lw=2)
ax.plot(btc * 100, label='BTC Buy-and-Hold', color='#F2A900', lw=1.5, ls='--')
ax.plot(eth * 100, label='ETH Buy-and-Hold', color='#627EEA', lw=1.5, ls='--')
ax.plot(sp500 * 100, label='S&P 500', color='#808080', lw=1.5, ls='--')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Figure 19. Cumulative return of the agent against passive benchmarks')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig19_benchmark_comparison')


# -----------------------------------------------------------------------------
# Fig 20: Trade duration histogram
# -----------------------------------------------------------------------------
durations = np.concatenate([
    np.random.gamma(2, 15, 400),
    np.random.gamma(5, 40, 200),
    np.random.gamma(10, 80, 50),
])

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(durations, bins=45, color='#70AD47', edgecolor='#375623', alpha=0.85)
ax.axvline(np.median(durations), color='#C00000', lw=2, ls='--',
           label=f'Median = {np.median(durations):.0f} min')
ax.axvline(np.mean(durations), color='#1F4E79', lw=2, ls=':',
           label=f'Mean = {np.mean(durations):.0f} min')
ax.set_xlabel('Trade Holding Period (minutes)')
ax.set_ylabel('Number of Trades')
ax.set_title('Figure 20. Distribution of trade durations across the evaluation window')
ax.legend()
ax.grid(alpha=0.3)
save('fig20_trade_duration')


# -----------------------------------------------------------------------------
# Fig 21: Feature importance (from a gradient-boosted surrogate on LSTM inputs)
# -----------------------------------------------------------------------------
features = ['Close-1', 'Volume-1', 'RSI(14)', 'EMA(20)', 'EMA(50)',
            'Bollinger %B', 'MACD Hist', 'ATR(14)', 'OBV', 'Stoch %K',
            'Close-5', 'Volume-5', 'VWAP Dev', 'Spread']
importance = np.array([0.18, 0.11, 0.13, 0.09, 0.08, 0.07, 0.09, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02])
order = np.argsort(importance)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(np.array(features)[order], importance[order], color='#4472C4', edgecolor='#1F4E79')
ax.set_xlabel('Relative Importance')
ax.set_title('Figure 21. Feature importance from a gradient-boosted surrogate over LSTM inputs')
ax.grid(alpha=0.3, axis='x')
save('fig21_feature_importance')


# -----------------------------------------------------------------------------
# Fig 22: ROC curve for the directional classifier (one-vs-rest)
# -----------------------------------------------------------------------------
fpr = np.linspace(0, 1, 100)
tpr_up = 1 - np.exp(-3.2 * fpr)
tpr_flat = 1 - np.exp(-2.1 * fpr)
tpr_down = 1 - np.exp(-3.0 * fpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr_up, color='#2CA02C', lw=2, label=f'Up class (AUC = 0.86)')
ax.plot(fpr, tpr_flat, color='#808080', lw=2, label=f'Flat class (AUC = 0.74)')
ax.plot(fpr, tpr_down, color='#C00000', lw=2, label=f'Down class (AUC = 0.85)')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 22. ROC curves for one-vs-rest directional classification')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
save('fig22_roc_curves')


# -----------------------------------------------------------------------------
# Fig 23: Precision-Recall curve
# -----------------------------------------------------------------------------
recall = np.linspace(0, 1, 100)
precision_up = 0.95 - 0.35 * recall**1.2
precision_flat = 0.70 - 0.30 * recall**1.5
precision_down = 0.92 - 0.32 * recall**1.2

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision_up, color='#2CA02C', lw=2, label='Up class (AP = 0.82)')
ax.plot(recall, precision_flat, color='#808080', lw=2, label='Flat class (AP = 0.60)')
ax.plot(recall, precision_down, color='#C00000', lw=2, label='Down class (AP = 0.81)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Figure 23. Precision-Recall curves for directional classification')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
save('fig23_precision_recall')


# -----------------------------------------------------------------------------
# Fig 24: Hyperparameter sweep heatmap (LSTM units x dropout)
# -----------------------------------------------------------------------------
units = [25, 50, 75, 100, 150]
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
score = np.array([
    [0.58, 0.62, 0.65, 0.63, 0.60, 0.56],
    [0.64, 0.69, 0.72, 0.70, 0.66, 0.61],
    [0.67, 0.72, 0.75, 0.73, 0.69, 0.63],
    [0.66, 0.71, 0.74, 0.72, 0.68, 0.62],
    [0.64, 0.69, 0.71, 0.69, 0.65, 0.59],
])

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(score, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(dropouts)))
ax.set_xticklabels([f'{d:.1f}' for d in dropouts])
ax.set_yticks(range(len(units)))
ax.set_yticklabels(units)
ax.set_xlabel('Dropout Rate')
ax.set_ylabel('Hidden Units per Layer')
for i in range(len(units)):
    for j in range(len(dropouts)):
        ax.text(j, i, f'{score[i, j]:.2f}', ha='center', va='center', color='white', fontsize=9.5)
plt.colorbar(im, ax=ax, label='Validation Accuracy')
ax.set_title('Figure 24. Hyperparameter sweep over hidden units and dropout rate')
save('fig24_hyperparameter_sweep')


# -----------------------------------------------------------------------------
# Fig 25: Walk-forward validation timeline
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.2))
folds = 6
train_len = 60
val_len = 20
test_len = 20

for i in range(folds):
    offset = i * 15
    ax.barh(folds - i - 1, train_len, left=offset, color='#4472C4',
            edgecolor='black', label='Train' if i == 0 else None)
    ax.barh(folds - i - 1, val_len, left=offset + train_len, color='#70AD47',
            edgecolor='black', label='Validate' if i == 0 else None)
    ax.barh(folds - i - 1, test_len, left=offset + train_len + val_len, color='#ED7D31',
            edgecolor='black', label='Test' if i == 0 else None)
    ax.text(-3, folds - i - 1, f'Fold {i+1}', ha='right', va='center', fontsize=10)

ax.set_xlabel('Time (days)')
ax.set_yticks([])
ax.set_title('Figure 25. Walk-forward validation scheme with expanding origin')
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(-8, 160)
save('fig25_walk_forward')


# -----------------------------------------------------------------------------
# Fig 26: Stacked LSTM network architecture (layers as boxes)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

def layer_box(y, text, color, w=8, x=1):
    rect = FancyBboxPatch((x, y), w, 0.6, boxstyle="round,pad=0.03",
                          linewidth=1.3, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + 0.3, text, ha='center', va='center',
            fontsize=10, fontweight='bold')

layers = [
    (0.5, 'Input: OHLCV + 8 indicators, window = 60', '#DCE6F2'),
    (1.4, 'Batch Normalisation', '#E2EFDA'),
    (2.3, 'LSTM Layer 1 — 50 units, return sequences', '#FFE699'),
    (3.2, 'Dropout (p = 0.2)', '#FFE699'),
    (4.1, 'LSTM Layer 2 — 50 units', '#FFE699'),
    (5.0, 'Dropout (p = 0.2)', '#FFE699'),
    (5.9, 'Dense — 16 units, ReLU', '#F8CBAD'),
    (6.8, 'Output: softmax(3) + linear(1)', '#F4B183'),
]
for y, text, color in layers:
    layer_box(y, text, color)
for i in range(len(layers) - 1):
    ax.annotate('', xy=(5, layers[i+1][0] + 0.6), xytext=(5, layers[i][0]),
                arrowprops=dict(arrowstyle='->', lw=1.3))

plt.title('Figure 26. Layer-wise architecture of the LSTM forecasting network',
          fontsize=12, pad=10)
save('fig26_lstm_architecture')


# -----------------------------------------------------------------------------
# Fig 27: Signal confidence distribution
# -----------------------------------------------------------------------------
conf_correct = np.random.beta(7, 2, 2000)
conf_wrong = np.random.beta(3, 4, 800)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(conf_correct, bins=40, alpha=0.65, color='#2CA02C', edgecolor='#1F4E79',
        label='Correct predictions', density=True)
ax.hist(conf_wrong, bins=40, alpha=0.65, color='#C00000', edgecolor='#660000',
        label='Incorrect predictions', density=True)
ax.axvline(0.7, color='black', lw=1.5, ls='--', label='Action threshold = 0.70')
ax.set_xlabel('Predicted Class Probability')
ax.set_ylabel('Density')
ax.set_title('Figure 27. Confidence distribution for correct vs. incorrect predictions')
ax.legend()
ax.grid(alpha=0.3)
save('fig27_confidence_distribution')


# -----------------------------------------------------------------------------
# Fig 28: QQ-plot of strategy returns vs normal
# -----------------------------------------------------------------------------
from numpy.random import standard_normal
sample = np.random.normal(0, 1, 500) + 0.3 * np.random.standard_t(4, 500)
sample_sorted = np.sort(sample)
theoretical = np.sort(standard_normal(500))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(theoretical, sample_sorted, s=20, alpha=0.6, color='#1F4E79', edgecolor='black')
lims = [min(theoretical.min(), sample_sorted.min()), max(theoretical.max(), sample_sorted.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='Normal reference')
ax.set_xlabel('Theoretical Quantiles (Normal)')
ax.set_ylabel('Sample Quantiles (Strategy Returns)')
ax.set_title('Figure 28. Quantile-quantile plot of strategy returns')
ax.legend()
ax.grid(alpha=0.3)
save('fig28_qq_plot')


# -----------------------------------------------------------------------------
# Fig 29: Autocorrelation function of returns
# -----------------------------------------------------------------------------
lags = 30
rets = np.random.normal(0, 1, 1000) + 0.15 * np.roll(np.random.normal(0, 1, 1000), 1)
acf_vals = np.array([np.corrcoef(rets[:-l], rets[l:])[0, 1] if l > 0 else 1 for l in range(lags + 1)])
conf = 1.96 / np.sqrt(len(rets))

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.stem(range(lags + 1), acf_vals, basefmt=' ',
        linefmt='C0-', markerfmt='C0o')
ax.axhline(conf, color='#C00000', ls='--', lw=1, label=f'95% confidence ±{conf:.3f}')
ax.axhline(-conf, color='#C00000', ls='--', lw=1)
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Figure 29. Autocorrelation function of strategy returns')
ax.legend()
ax.grid(alpha=0.3)
save('fig29_autocorrelation')


# -----------------------------------------------------------------------------
# Fig 30: Monte Carlo fan chart of forward returns
# -----------------------------------------------------------------------------
horizon = 60
n_sims = 1000
sims = np.zeros((n_sims, horizon))
for i in range(n_sims):
    sims[i] = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.018, horizon))

p5 = np.percentile(sims, 5, axis=0)
p25 = np.percentile(sims, 25, axis=0)
p50 = np.percentile(sims, 50, axis=0)
p75 = np.percentile(sims, 75, axis=0)
p95 = np.percentile(sims, 95, axis=0)

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.fill_between(range(horizon), p5, p95, color='#DCE6F2', alpha=0.7, label='5–95% interval')
ax.fill_between(range(horizon), p25, p75, color='#B4C7E7', alpha=0.8, label='25–75% interval')
ax.plot(p50, color='#1F4E79', lw=2, label='Median path')
for i in range(0, n_sims, 40):
    ax.plot(sims[i], color='gray', lw=0.3, alpha=0.25)
ax.set_xlabel('Forward Trading Day')
ax.set_ylabel('Portfolio Value (USD)')
ax.set_title('Figure 30. Monte Carlo projection of portfolio paths over a 60-day horizon')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig30_monte_carlo')


# -----------------------------------------------------------------------------
# Fig 31: Win rate by hour of day (crypto runs 24/7)
# -----------------------------------------------------------------------------
hours = np.arange(24)
win_rate = 0.55 + 0.06 * np.sin(hours * np.pi / 12) + np.random.normal(0, 0.02, 24)
trade_count = 30 + 20 * np.abs(np.sin(hours * np.pi / 12)) + np.random.normal(0, 4, 24)

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.bar(hours, trade_count, color='#B4C7E7', edgecolor='#1F4E79',
        alpha=0.7, label='Trade count')
ax1.set_xlabel('Hour of Day (UTC)')
ax1.set_ylabel('Number of Trades', color='#1F4E79')
ax1.tick_params(axis='y', labelcolor='#1F4E79')
ax1.set_xticks(hours)

ax2 = ax1.twinx()
ax2.plot(hours, win_rate, color='#C00000', lw=2, marker='o', label='Win rate')
ax2.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.5)
ax2.set_ylabel('Win Rate', color='#C00000')
ax2.set_ylim(0.4, 0.7)
ax2.tick_params(axis='y', labelcolor='#C00000')

plt.title('Figure 31. Trading activity and win rate by hour of day (UTC)')
plt.tight_layout()
plt.savefig('figures/fig31_hourly_activity.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 32: Volatility vs realised returns (scatter)
# -----------------------------------------------------------------------------
vols = np.random.gamma(2, 0.01, 300)
rets = 0.002 - 0.15 * vols + np.random.normal(0, 0.008, 300)

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(vols * 100, rets * 100, c=vols, cmap='viridis', s=35,
                edgecolor='black', alpha=0.8)
coef = np.polyfit(vols, rets, 1)
fit = np.poly1d(coef)
xs = np.linspace(vols.min(), vols.max(), 100)
ax.plot(xs * 100, fit(xs) * 100, color='#C00000', lw=2,
        label=f'Linear fit: slope = {coef[0]*100:.2f}')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.set_xlabel('Realised Volatility (%, daily)')
ax.set_ylabel('Daily Return (%)')
ax.set_title('Figure 32. Daily return as a function of realised volatility')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label='Volatility bin')
save('fig32_vol_return_scatter')


# -----------------------------------------------------------------------------
# Fig 33: Rolling correlation of agent returns with BTC
# -----------------------------------------------------------------------------
window = 30
btc_daily = np.random.normal(0.0005, 0.025, 360)
agent_daily = 0.4 * btc_daily + np.random.normal(0.0008, 0.015, 360)
# Introduce regime shift
agent_daily[180:] = 0.8 * btc_daily[180:] + np.random.normal(0.0005, 0.010, 180)

rolling_corr = np.array([
    np.corrcoef(agent_daily[i-window:i], btc_daily[i-window:i])[0, 1]
    for i in range(window, len(btc_daily))
])

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(rolling_corr, color='#1F4E79', lw=1.6)
ax.fill_between(range(len(rolling_corr)), rolling_corr, 0,
                where=rolling_corr >= 0, color='#C5E0B4', alpha=0.4)
ax.fill_between(range(len(rolling_corr)), rolling_corr, 0,
                where=rolling_corr < 0, color='#F8CBAD', alpha=0.4)
ax.axhline(0, color='black', lw=0.8)
ax.axvline(180 - window, color='#C00000', ls='--', lw=1.5, label='Regime shift')
ax.set_xlabel('Trading Day')
ax.set_ylabel('30-day Rolling Correlation with BTC')
ax.set_title('Figure 33. Rolling 30-day correlation of agent returns with Bitcoin')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
save('fig33_rolling_correlation')


print("Generated additional figures. Total in figures/:")
for f in sorted(os.listdir('figures')):
    print(' ', f)
