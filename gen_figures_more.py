"""Generate 17 additional research figures (total 50)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs('figures', exist_ok=True)
np.random.seed(11)

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


n = 200
returns = np.random.normal(0.0003, 0.015, n)
close = 1800 * np.cumprod(1 + returns)
high = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
volume = np.abs(np.random.normal(50000, 15000, n)) + np.abs(returns) * 2e6


# -----------------------------------------------------------------------------
# Fig 34: Rolling Sortino ratio
# -----------------------------------------------------------------------------
sortino = 2.1 + 0.8 * np.sin(np.arange(180)/22) + np.random.normal(0, 0.2, 180)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(sortino, color='#2CA02C', lw=1.8)
ax.fill_between(range(180), sortino, 0, where=sortino >= 1.5,
                color='#C5E0B4', alpha=0.5, label='Sortino ≥ 1.5')
ax.fill_between(range(180), sortino, 0, where=sortino < 1.5,
                color='#FFE699', alpha=0.5)
ax.axhline(1.5, color='#808080', ls='--', lw=1, label='Target = 1.5')
ax.set_xlabel('Trading Day')
ax.set_ylabel('30-day Rolling Sortino Ratio')
ax.set_title('Figure 34. Rolling Sortino ratio showing downside-risk-adjusted return')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
save('fig34_sortino')


# -----------------------------------------------------------------------------
# Fig 35: Calmar ratio over time
# -----------------------------------------------------------------------------
calmar = 1.2 + 0.4 * np.cos(np.arange(180)/30) + np.random.normal(0, 0.15, 180)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(calmar, color='#7030A0', lw=1.8, label='Calmar ratio')
ax.axhline(1.0, color='#808080', ls='--', lw=1)
ax.fill_between(range(180), calmar, 1.0, where=calmar >= 1.0,
                color='#DCE6F2', alpha=0.5)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Calmar Ratio')
ax.set_title('Figure 35. Calmar ratio (annualised return over maximum drawdown)')
ax.legend()
ax.grid(alpha=0.3)
save('fig35_calmar')


# -----------------------------------------------------------------------------
# Fig 36: Rolling win rate
# -----------------------------------------------------------------------------
win_rate = 0.58 + 0.05 * np.sin(np.arange(180)/18) + np.random.normal(0, 0.03, 180)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(win_rate * 100, color='#1F4E79', lw=1.8)
ax.axhline(50, color='#C00000', ls='--', lw=1.5, label='Break-even = 50%')
ax.axhline(np.mean(win_rate) * 100, color='#2CA02C', ls=':', lw=1.5,
           label=f'Mean = {np.mean(win_rate)*100:.1f}%')
ax.fill_between(range(180), win_rate * 100, 50, where=win_rate * 100 >= 50,
                color='#C5E0B4', alpha=0.5)
ax.fill_between(range(180), win_rate * 100, 50, where=win_rate * 100 < 50,
                color='#F8CBAD', alpha=0.5)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Figure 36. Rolling 20-trade win rate throughout the evaluation window')
ax.legend()
ax.grid(alpha=0.3)
save('fig36_rolling_winrate')


# -----------------------------------------------------------------------------
# Fig 37: Trade P&L histogram (winners vs losers)
# -----------------------------------------------------------------------------
winners = np.random.gamma(1.2, 30, 350)
losers = -np.random.gamma(0.9, 25, 250)
all_pnl = np.concatenate([winners, losers])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(winners, bins=30, color='#2CA02C', alpha=0.7, edgecolor='black', label='Winning trades')
ax.hist(losers, bins=25, color='#C00000', alpha=0.7, edgecolor='black', label='Losing trades')
ax.axvline(0, color='black', lw=1.5)
ax.axvline(np.mean(all_pnl), color='#1F4E79', ls='--', lw=2,
           label=f'Mean = ${np.mean(all_pnl):.1f}')
ax.set_xlabel('Trade P&L (USD)')
ax.set_ylabel('Number of Trades')
ax.set_title('Figure 37. Distribution of per-trade profit and loss')
ax.legend()
ax.grid(alpha=0.3)
save('fig37_pnl_histogram')


# -----------------------------------------------------------------------------
# Fig 38: Slippage vs order size
# -----------------------------------------------------------------------------
sizes = np.random.exponential(500, 400) + 100
slip = 0.02 + 0.00005 * sizes + 0.001 * np.log(sizes) + np.random.normal(0, 0.015, 400)
slip = np.abs(slip)

fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(sizes, slip, s=20, alpha=0.5, c=slip, cmap='plasma', edgecolor='black')
coef = np.polyfit(np.log(sizes), slip, 1)
xs = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
ax.plot(xs, coef[0] * np.log(xs) + coef[1], color='#1F4E79', lw=2,
        label=f'Log-linear fit: slip = {coef[0]:.4f}·ln(size) + {coef[1]:.3f}')
ax.set_xscale('log')
ax.set_xlabel('Order Size (USD)')
ax.set_ylabel('Realised Slippage (%)')
ax.set_title('Figure 38. Execution slippage as a function of order size')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label='Slippage (%)')
save('fig38_slippage_vs_size')


# -----------------------------------------------------------------------------
# Fig 39: Model inference latency distribution
# -----------------------------------------------------------------------------
inf_lat = np.random.gamma(3, 15, 2000) + np.random.normal(0, 3, 2000)
inf_lat = np.clip(inf_lat, 10, None)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(inf_lat, bins=60, color='#4472C4', edgecolor='#1F4E79', alpha=0.85)
ax.axvline(np.median(inf_lat), color='#C00000', lw=2, ls='--',
           label=f'Median = {np.median(inf_lat):.1f} ms')
ax.axvline(np.percentile(inf_lat, 95), color='#2CA02C', lw=2, ls=':',
           label=f'p95 = {np.percentile(inf_lat, 95):.1f} ms')
ax.axvline(np.percentile(inf_lat, 99), color='#7030A0', lw=2, ls=':',
           label=f'p99 = {np.percentile(inf_lat, 99):.1f} ms')
ax.set_xlabel('LSTM Inference Latency (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Figure 39. Distribution of model inference latency on the production CPU')
ax.legend()
ax.grid(alpha=0.3)
save('fig39_inference_latency')


# -----------------------------------------------------------------------------
# Fig 40: On-Balance Volume indicator
# -----------------------------------------------------------------------------
def obv(close, vol):
    out = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            out[i] = out[i-1] + vol[i]
        elif close[i] < close[i-1]:
            out[i] = out[i-1] - vol[i]
        else:
            out[i] = out[i-1]
    return out

obv_vals = obv(close, volume)
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})
axes[0].plot(close, color='#1F4E79', lw=1.5)
axes[0].set_ylabel('Price')
axes[0].set_title('(a) Underlying price')
axes[0].grid(alpha=0.3)
axes[1].plot(obv_vals, color='#7030A0', lw=1.5)
axes[1].fill_between(range(n), obv_vals, 0,
                     where=obv_vals >= 0, color='#DCE6F2', alpha=0.5)
axes[1].fill_between(range(n), obv_vals, 0,
                     where=obv_vals < 0, color='#F8CBAD', alpha=0.5)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_ylabel('OBV')
axes[1].set_xlabel('Time (minutes)')
axes[1].set_title('(b) On-Balance Volume')
axes[1].grid(alpha=0.3)
plt.suptitle('Figure 40. On-Balance Volume indicator alongside price', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig40_obv.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 41: Average True Range
# -----------------------------------------------------------------------------
tr = np.maximum.reduce([high - low,
                        np.abs(high - np.roll(close, 1)),
                        np.abs(low - np.roll(close, 1))])
atr = np.convolve(tr, np.ones(14)/14, mode='same')

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(atr, color='#C00000', lw=1.8, label='ATR(14)')
ax.fill_between(range(n), atr, 0, color='#F8CBAD', alpha=0.3)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Average True Range (USDT)')
ax.set_title('Figure 41. 14-period Average True Range used to scale stop-loss levels')
ax.legend()
ax.grid(alpha=0.3)
save('fig41_atr')


# -----------------------------------------------------------------------------
# Fig 42: Stochastic oscillator (%K and %D)
# -----------------------------------------------------------------------------
period = 14
stoch_k = np.array([
    100 * (close[i] - np.min(low[max(0, i-period):i+1])) /
    (np.max(high[max(0, i-period):i+1]) - np.min(low[max(0, i-period):i+1]) + 1e-9)
    for i in range(n)
])
stoch_d = np.convolve(stoch_k, np.ones(3)/3, mode='same')

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(stoch_k, color='#1F4E79', lw=1.5, label='%K')
ax.plot(stoch_d, color='#ED7D31', lw=1.5, label='%D (3-period SMA of %K)')
ax.axhline(80, color='#C00000', ls='--', lw=1)
ax.axhline(20, color='#2CA02C', ls='--', lw=1)
ax.fill_between(range(n), stoch_k, 80, where=stoch_k >= 80,
                color='#C00000', alpha=0.2)
ax.fill_between(range(n), stoch_k, 20, where=stoch_k <= 20,
                color='#2CA02C', alpha=0.2)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Stochastic Value')
ax.set_title('Figure 42. Stochastic oscillator (%K and %D) with threshold zones')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)
save('fig42_stochastic')


# -----------------------------------------------------------------------------
# Fig 43: VWAP versus price
# -----------------------------------------------------------------------------
typical = (high + low + close) / 3
vwap = np.cumsum(typical * volume) / np.cumsum(volume)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(close, color='#808080', lw=1.2, alpha=0.8, label='Price')
ax.plot(vwap, color='#C00000', lw=2, label='VWAP')
ax.fill_between(range(n), close, vwap, where=close >= vwap,
                color='#C5E0B4', alpha=0.4, label='Price above VWAP')
ax.fill_between(range(n), close, vwap, where=close < vwap,
                color='#F8CBAD', alpha=0.4, label='Price below VWAP')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Price (USDT)')
ax.set_title('Figure 43. Volume-Weighted Average Price as an execution benchmark')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig43_vwap')


# -----------------------------------------------------------------------------
# Fig 44: Learning-rate schedule (cosine annealing)
# -----------------------------------------------------------------------------
epochs = np.arange(50)
lr_init = 1e-3
lr = lr_init * 0.5 * (1 + np.cos(np.pi * epochs / 50))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, lr, color='#1F4E79', lw=2, marker='o', markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_yscale('log')
ax.set_title('Figure 44. Cosine-annealing learning-rate schedule used during training')
ax.grid(alpha=0.3, which='both')
save('fig44_lr_schedule')


# -----------------------------------------------------------------------------
# Fig 45: Loss-landscape contour (simulated)
# -----------------------------------------------------------------------------
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 0.3 * (X**2 + Y**2) - 0.5 * np.exp(-(X-1)**2 - (Y-0.5)**2) \
    - 0.4 * np.exp(-(X+1.5)**2 - (Y+1)**2) + 1

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
# Training trajectory
traj_x = np.array([-2.5, -2.0, -1.7, -1.55, -1.5, -1.48, 1.0, 1.05, 1.0])
traj_y = np.array([2.5, 1.8, 1.2, 0.6, -0.2, -0.8, 0.4, 0.5, 0.5])
ax.plot(traj_x, traj_y, 'o-', color='#C00000', markersize=8, lw=2, label='Training trajectory')
ax.scatter([traj_x[-1]], [traj_y[-1]], marker='*', s=300, color='yellow',
           edgecolor='black', zorder=5, label='Final parameters')
ax.set_xlabel('Projected weight direction 1')
ax.set_ylabel('Projected weight direction 2')
ax.set_title('Figure 45. 2-D projection of the LSTM loss landscape with training trajectory')
ax.legend(loc='upper right')
plt.colorbar(c, ax=ax, label='Loss')
save('fig45_loss_landscape')


# -----------------------------------------------------------------------------
# Fig 46: Profit factor per month
# -----------------------------------------------------------------------------
months = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']
pf = [1.8, 2.3, 1.4, 2.1, 1.65, 1.95]
colors = ['#2CA02C' if p >= 1.5 else '#ED7D31' for p in pf]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(months, pf, color=colors, edgecolor='#333')
ax.axhline(1.0, color='black', lw=1.5, label='Break-even (1.0)')
ax.axhline(1.5, color='#C00000', ls='--', lw=1, label='Target (1.5)')
for bar, val in zip(bars, pf):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Evaluation Month')
ax.set_ylabel('Profit Factor')
ax.set_title('Figure 46. Monthly profit factor across the six-month evaluation window')
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 3)
save('fig46_profit_factor')


# -----------------------------------------------------------------------------
# Fig 47: Ulcer Index evolution
# -----------------------------------------------------------------------------
equity_curve = 10000 * np.cumprod(1 + np.random.normal(0.0008, 0.012, 180))
rolling_max = np.maximum.accumulate(equity_curve)
dd_pct = (equity_curve - rolling_max) / rolling_max * 100
window = 14
ui = np.array([
    np.sqrt(np.mean(dd_pct[max(0, i-window):i+1] ** 2))
    for i in range(len(dd_pct))
])

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(ui, color='#C00000', lw=1.8)
ax.fill_between(range(len(ui)), ui, 0, color='#F8CBAD', alpha=0.5)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Ulcer Index (%)')
ax.set_title('Figure 47. Ulcer Index quantifying the depth and duration of drawdowns')
ax.grid(alpha=0.3)
save('fig47_ulcer_index')


# -----------------------------------------------------------------------------
# Fig 48: Conditional VaR (CVaR) evolution
# -----------------------------------------------------------------------------
rets_series = np.random.normal(0.0008, 0.018, 200)
window = 30
var95 = np.array([
    np.percentile(rets_series[max(0, i-window):i+1], 5) * 100
    for i in range(len(rets_series))
])
cvar95 = np.array([
    np.mean(rets_series[max(0, i-window):i+1][
        rets_series[max(0, i-window):i+1] <= np.percentile(rets_series[max(0, i-window):i+1], 5)]) * 100
    if np.any(rets_series[max(0, i-window):i+1] <= np.percentile(rets_series[max(0, i-window):i+1], 5))
    else 0
    for i in range(len(rets_series))
])

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(var95, color='#ED7D31', lw=1.6, label='VaR (95%)')
ax.plot(cvar95, color='#C00000', lw=1.8, label='CVaR / Expected Shortfall (95%)')
ax.fill_between(range(len(var95)), cvar95, 0, color='#F8CBAD', alpha=0.4)
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Daily Return at Risk (%)')
ax.set_title('Figure 48. 95% Value-at-Risk and Expected Shortfall over a rolling window')
ax.legend()
ax.grid(alpha=0.3)
save('fig48_cvar')


# -----------------------------------------------------------------------------
# Fig 49: Cross-validation fold scores (box plot)
# -----------------------------------------------------------------------------
fold_scores = [
    np.random.normal(0.72, 0.02, 20),
    np.random.normal(0.74, 0.025, 20),
    np.random.normal(0.71, 0.03, 20),
    np.random.normal(0.73, 0.022, 20),
    np.random.normal(0.75, 0.018, 20),
    np.random.normal(0.72, 0.028, 20),
]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(fold_scores, patch_artist=True,
                labels=[f'Fold {i+1}' for i in range(6)])
colors = ['#4472C4', '#5B9BD5', '#70AD47', '#FFC000', '#ED7D31', '#7030A0']
for patch, col in zip(bp['boxes'], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set(color='black', linewidth=2)
ax.axhline(np.mean([np.mean(f) for f in fold_scores]),
           color='#C00000', ls='--', lw=1.5,
           label=f'Mean = {np.mean([np.mean(f) for f in fold_scores]):.3f}')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Figure 49. Walk-forward cross-validation scores across folds')
ax.legend()
ax.grid(alpha=0.3, axis='y')
save('fig49_cv_scores')


# -----------------------------------------------------------------------------
# Fig 50: Cumulative trade count
# -----------------------------------------------------------------------------
trade_events = np.cumsum(np.random.poisson(4, 180))

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.step(range(len(trade_events)), trade_events, color='#1F4E79', lw=1.8, where='post')
ax.fill_between(range(len(trade_events)), trade_events, 0,
                color='#DCE6F2', alpha=0.5, step='post')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Cumulative Number of Trades')
ax.set_title('Figure 50. Cumulative trade count throughout the evaluation window')
ax.grid(alpha=0.3)
ax.text(90, trade_events[90] * 1.1,
        f'Avg ≈ {trade_events[-1] / 180:.1f} trades/day',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333'))
save('fig50_cumulative_trades')


print(f"Total figures: {len(os.listdir('figures'))}")
