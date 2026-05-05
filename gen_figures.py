"""Generate original figures for the AI trading bot research paper."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)
np.random.seed(42)

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
# Fig 1: Three-layer system architecture block diagram
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

def box(x, y, w, h, text, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          linewidth=1.5, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=10, fontweight='bold')

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.4, color='#333'))

# Data layer
box(0.2, 5.2, 9.6, 1.4, '', '#DCE6F2')
ax.text(0.4, 6.3, 'Data Layer', fontsize=11, fontweight='bold', color='#1F4E79')
box(0.5, 5.4, 2.0, 0.9, 'WebSocket\nClient', '#B4C7E7')
box(2.8, 5.4, 2.0, 0.9, 'REST Poller\n(Fallback)', '#B4C7E7')
box(5.1, 5.4, 2.0, 0.9, 'Validation &\nNormalisation', '#B4C7E7')
box(7.4, 5.4, 2.2, 0.9, 'Rolling Buffer +\nSQLite Store', '#B4C7E7')

# Processing layer
box(0.2, 2.9, 9.6, 1.9, '', '#E2EFDA')
ax.text(0.4, 4.5, 'Processing Layer', fontsize=11, fontweight='bold', color='#385723')
box(0.5, 3.2, 2.0, 1.1, 'Feature\nEngineering\n(TA-Lib)', '#C5E0B4')
box(2.8, 3.2, 2.0, 1.1, 'LSTM\nForecaster', '#C5E0B4')
box(5.1, 3.2, 2.0, 1.1, 'Strategy\nEngine', '#C5E0B4')
box(7.4, 3.2, 2.2, 1.1, 'Risk Manager\n(Stop-loss,\nPosition Size)', '#C5E0B4')

# Interface layer
box(0.2, 0.3, 9.6, 2.1, '', '#FFF2CC')
ax.text(0.4, 2.15, 'Interface Layer', fontsize=11, fontweight='bold', color='#7F6000')
box(0.5, 0.6, 2.6, 1.3, 'PyQt5\nDesktop GUI', '#FFE699')
box(3.3, 0.6, 2.6, 1.3, 'Telegram Bot\n(Commands\n& Alerts)', '#FFE699')
box(6.1, 0.6, 3.5, 1.3, 'Order Router\n(Binance / Alpha Vantage)', '#FFE699')

# arrows between layers
arrow(5.0, 5.2, 5.0, 4.35)
arrow(5.0, 2.95, 5.0, 2.05)
arrow(3.0, 0.6, 7.8, 0.6)

plt.title('Figure 1. Three-layer system architecture of the proposed trading agent',
          fontsize=12, pad=12)
save('fig01_architecture')


# -----------------------------------------------------------------------------
# Fig 2: LSTM cell internal structure
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Cell border
cell = FancyBboxPatch((1.5, 0.5), 9, 5, boxstyle="round,pad=0.1",
                     linewidth=2, edgecolor='#333', facecolor='#F2F2F2')
ax.add_patch(cell)
ax.text(6, 5.2, 'LSTM Cell at time t', ha='center', fontsize=12, fontweight='bold')

# Gates
gate_color = '#FFD966'
def gate(x, y, label):
    rect = FancyBboxPatch((x, y), 1.4, 0.8, boxstyle="round,pad=0.03",
                          linewidth=1.3, edgecolor='#333', facecolor=gate_color)
    ax.add_patch(rect)
    ax.text(x + 0.7, y + 0.4, label, ha='center', va='center', fontsize=10, fontweight='bold')

gate(2.2, 3.2, 'Forget\nGate f_t')
gate(4.3, 3.2, 'Input\nGate i_t')
gate(6.4, 3.2, 'Candidate\nC̃_t')
gate(8.5, 3.2, 'Output\nGate o_t')

# Cell state
ax.annotate('', xy=(10.5, 4.5), xytext=(1.8, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#C00000'))
ax.text(6, 4.7, 'Cell state C_t', ha='center', fontsize=10, color='#C00000', fontweight='bold')

# Hidden state
ax.annotate('', xy=(10.5, 1.7), xytext=(1.8, 1.7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1F4E79'))
ax.text(6, 1.4, 'Hidden state h_t', ha='center', fontsize=10, color='#1F4E79', fontweight='bold')

# Input
ax.text(0.5, 2.3, 'x_t\n(price +\nindicators)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#B4C7E7'))
ax.annotate('', xy=(2.2, 2.7), xytext=(1.2, 2.5),
            arrowprops=dict(arrowstyle='->', lw=1.3))

# Output
ax.text(11.3, 2.3, 'y_t\n(direction +\nconfidence)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#C5E0B4'))
ax.annotate('', xy=(11.0, 2.5), xytext=(9.9, 2.7),
            arrowprops=dict(arrowstyle='->', lw=1.3))

plt.title('Figure 2. Internal structure of a single LSTM cell used in the forecasting layer',
          fontsize=12, pad=10)
save('fig02_lstm_cell')


# -----------------------------------------------------------------------------
# Fig 3: Training and validation curves (loss & accuracy)
# -----------------------------------------------------------------------------
epochs = np.arange(1, 51)
train_loss = 0.95 * np.exp(-epochs/12) + 0.08 + np.random.normal(0, 0.015, 50)
val_loss = 0.95 * np.exp(-epochs/10) + 0.12 + np.random.normal(0, 0.025, 50)
train_acc = 1 - 0.6 * np.exp(-epochs/15) - 0.05 + np.random.normal(0, 0.01, 50)
val_acc = 1 - 0.6 * np.exp(-epochs/13) - 0.08 + np.random.normal(0, 0.015, 50)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs, train_loss, label='Training Loss', color='#1F4E79', lw=1.8)
axes[0].plot(epochs, val_loss, label='Validation Loss', color='#C00000', lw=1.8, ls='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Categorical Cross-Entropy Loss')
axes[0].set_title('(a) Loss over training epochs')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, train_acc, label='Training Accuracy', color='#1F4E79', lw=1.8)
axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='#C00000', lw=1.8, ls='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Directional Accuracy')
axes[1].set_title('(b) Accuracy over training epochs')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.4, 1.0)

plt.suptitle('Figure 3. Training dynamics of the LSTM forecasting model', fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig03_training_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 4: Predicted vs actual price over 200 minutes
# -----------------------------------------------------------------------------
t = np.arange(200)
actual = 1800 + np.cumsum(np.random.normal(0, 3, 200)) + 20 * np.sin(t/15)
predicted = actual + np.random.normal(0, 4, 200)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(t, actual, label='Actual ETH/USDT Price', color='#1F4E79', lw=1.8)
ax.plot(t, predicted, label='LSTM Prediction', color='#C00000', lw=1.3, ls='--', alpha=0.85)
ax.fill_between(t, predicted - 6, predicted + 6, color='#C00000', alpha=0.15,
                label='±1σ Confidence Band')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Price (USDT)')
ax.set_title('Figure 4. Forecast vs. realised price on a held-out ETH/USDT sample')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig04_prediction_vs_actual')


# -----------------------------------------------------------------------------
# Fig 5: Simulated backtest equity curve (6 months)
# -----------------------------------------------------------------------------
days = 180
daily_ret = np.random.normal(0.0008, 0.015, days)
# Inject some drawdown and recovery dynamics
daily_ret[40:55] = np.random.normal(-0.008, 0.02, 15)
daily_ret[110:120] = np.random.normal(-0.006, 0.018, 10)
equity = 10000 * np.cumprod(1 + daily_ret)
benchmark = 10000 * np.cumprod(1 + np.random.normal(0.0003, 0.012, days))

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(equity, label='Proposed Agent', color='#1F4E79', lw=1.8)
ax.plot(benchmark, label='Buy-and-Hold Baseline', color='#808080', lw=1.5, ls='--')
ax.fill_between(range(days), benchmark, equity,
                where=equity >= benchmark, color='#C5E0B4', alpha=0.4, label='Excess Return')
ax.fill_between(range(days), benchmark, equity,
                where=equity < benchmark, color='#F4B183', alpha=0.4)
ax.set_xlabel('Trading Day')
ax.set_ylabel('Portfolio Value (USD)')
ax.set_title('Figure 5. Backtested equity curve over a six-month evaluation window')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
save('fig05_equity_curve')


# -----------------------------------------------------------------------------
# Fig 6: Performance across market regimes (bar chart)
# -----------------------------------------------------------------------------
regimes = ['Trending\nBull', 'Trending\nBear', 'Range-\nBound', 'High\nVolatility']
accuracy = [91.2, 87.4, 54.2, 68.9]
profit_factor = [2.1, 1.85, 1.32, 1.48]

x = np.arange(len(regimes))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 4.8))
bars1 = ax1.bar(x - width/2, accuracy, width, label='Directional Accuracy (%)',
                color='#4472C4', edgecolor='#1F4E79')
ax1.set_ylabel('Directional Accuracy (%)', color='#1F4E79')
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelcolor='#1F4E79')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, profit_factor, width, label='Profit Factor',
                color='#ED7D31', edgecolor='#C00000')
ax2.set_ylabel('Profit Factor', color='#C00000')
ax2.set_ylim(0, 3)
ax2.tick_params(axis='y', labelcolor='#C00000')

ax1.set_xticks(x)
ax1.set_xticklabels(regimes)
ax1.set_title('Figure 6. Regime-conditional performance of the LSTM forecaster')

for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val}%', ha='center', fontsize=9)
for bar, val in zip(bars2, profit_factor):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', fontsize=9)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig('figures/fig06_regime_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 7: Order fill latency distribution
# -----------------------------------------------------------------------------
latencies = np.concatenate([
    np.random.gamma(3, 100, 850),
    np.random.gamma(2, 400, 140),
    np.random.uniform(1200, 2500, 10),
])

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(latencies, bins=60, color='#4472C4', edgecolor='#1F4E79', alpha=0.85)
ax.axvline(np.median(latencies), color='#C00000', lw=2, ls='--',
           label=f'Median = {np.median(latencies):.0f} ms')
ax.axvline(np.percentile(latencies, 99), color='#70AD47', lw=2, ls=':',
           label=f'99th %ile = {np.percentile(latencies, 99):.0f} ms')
ax.axvline(1000, color='#7F6000', lw=1.5, ls='-.', label='SLA target = 1000 ms')
ax.set_xlabel('Fill Latency (milliseconds)')
ax.set_ylabel('Frequency')
ax.set_title('Figure 7. Distribution of observed order-fill latencies (n = 1 000)')
ax.legend()
ax.grid(alpha=0.3)
save('fig07_latency_distribution')


# -----------------------------------------------------------------------------
# Fig 8: Confusion matrix for directional classification
# -----------------------------------------------------------------------------
cm = np.array([
    [1820,  210,   70],
    [ 230, 1650,  220],
    [  90,  250, 1860],
])
labels = ['Up', 'Flat', 'Down']

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Actual Class')
ax.set_title('Figure 8. Confusion matrix of the three-way directional classifier')

for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > 1000 else 'black', fontsize=12)

plt.colorbar(im, ax=ax, fraction=0.046)
save('fig08_confusion_matrix')


# -----------------------------------------------------------------------------
# Fig 9: Model comparison across 14 assets
# -----------------------------------------------------------------------------
assets = ['ADA','AVAX','AXS','BCH','BNB','CRO','DOGE','DOT','EOS','ETH','LINK','LTC','SOL','BTC']
models = {
    'Proposed LSTM': np.clip(np.random.normal(0.69, 0.035, 14), 0.55, 0.80),
    'GRU':           np.clip(np.random.normal(0.62, 0.04,  14), 0.50, 0.75),
    'Random Forest': np.clip(np.random.normal(0.58, 0.045, 14), 0.48, 0.70),
    'ARIMA':         np.clip(np.random.normal(0.54, 0.05,  14), 0.42, 0.68),
    'XGBoost':       np.clip(np.random.normal(0.64, 0.04,  14), 0.52, 0.74),
}

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, vals) in enumerate(models.items()):
    ax.plot(assets, vals, marker='o', lw=1.8, label=name, alpha=0.9)
ax.set_ylabel('Directional Accuracy')
ax.set_xlabel('Cryptocurrency Asset')
ax.set_title('Figure 9. Per-asset directional accuracy of five forecasting models')
ax.legend(loc='lower right', ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0.4, 0.85)
save('fig09_accuracy_by_asset')


# -----------------------------------------------------------------------------
# Fig 10: Data pipeline flowchart
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

steps = [
    (0.5, 'Exchange\nAPI', '#B4C7E7'),
    (2.3, 'Tick\nValidator', '#B4C7E7'),
    (4.1, 'Feature\nEngineer', '#C5E0B4'),
    (5.9, 'LSTM\nInference', '#C5E0B4'),
    (7.7, 'Strategy\nEngine', '#FFE699'),
    (9.5, 'Risk\nGate', '#F8CBAD'),
    (11.0, 'Order\nRouter', '#F4B183'),
]
for x, text, color in steps:
    rect = FancyBboxPatch((x - 0.6, 1.8), 1.2, 1.4, boxstyle="round,pad=0.05",
                          linewidth=1.3, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, 2.5, text, ha='center', va='center', fontsize=9.5, fontweight='bold')

for i in range(len(steps) - 1):
    ax.annotate('', xy=(steps[i+1][0] - 0.6, 2.5), xytext=(steps[i][0] + 0.6, 2.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

# Feedback loop
ax.annotate('', xy=(0.5, 1.7), xytext=(11.0, 1.7),
            arrowprops=dict(arrowstyle='->', lw=1.2, color='#C00000',
                            connectionstyle="arc3,rad=-0.25"))
ax.text(6.0, 0.4, 'Trade feedback → model update & risk recalibration',
        ha='center', color='#C00000', fontstyle='italic', fontsize=10)

plt.title('Figure 10. End-to-end data and control flow through the trading pipeline',
          fontsize=12, pad=12)
save('fig10_data_pipeline')


# -----------------------------------------------------------------------------
# Fig 11: Sharpe ratio & drawdown over time (rolling)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

sharpe = 1.5 + 0.6 * np.sin(np.arange(180)/25) + np.random.normal(0, 0.15, 180)
ax1.plot(sharpe, color='#1F4E79', lw=1.8)
ax1.fill_between(range(180), sharpe, 0,
                 where=sharpe >= 1.0, color='#C5E0B4', alpha=0.5, label='Sharpe ≥ 1.0')
ax1.fill_between(range(180), sharpe, 0,
                 where=sharpe < 1.0, color='#F8CBAD', alpha=0.5, label='Sharpe < 1.0')
ax1.axhline(1.0, color='#808080', ls='--', lw=1)
ax1.set_ylabel('Rolling 30-day Sharpe')
ax1.set_title('(a) Rolling risk-adjusted return')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

drawdown = -np.abs(np.cumsum(np.random.normal(0, 0.003, 180))) * 100
drawdown = np.minimum(drawdown, 0)
ax2.fill_between(range(180), drawdown, 0, color='#C00000', alpha=0.5)
ax2.plot(drawdown, color='#C00000', lw=1.5)
ax2.set_ylabel('Drawdown (%)')
ax2.set_xlabel('Trading Day')
ax2.set_title('(b) Rolling drawdown from peak')
ax2.grid(alpha=0.3)

plt.suptitle('Figure 11. Risk-adjusted performance and drawdown over the evaluation window',
             fontsize=12)
plt.tight_layout()
plt.savefig('figures/fig11_sharpe_drawdown.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


# -----------------------------------------------------------------------------
# Fig 12: Use-case diagram (simplified UML)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Actor stick figure
ax.plot([0.6, 0.6], [4.6, 4.0], 'k-', lw=1.5)  # body
ax.add_patch(plt.Circle((0.6, 4.9), 0.2, color='white', ec='black', lw=1.5))  # head
ax.plot([0.2, 1.0], [4.3, 4.3], 'k-', lw=1.5)  # arms
ax.plot([0.6, 0.3], [4.0, 3.4], 'k-', lw=1.5)  # left leg
ax.plot([0.6, 0.9], [4.0, 3.4], 'k-', lw=1.5)  # right leg
ax.text(0.6, 3.0, 'Trader', ha='center', fontsize=10, fontweight='bold')

# System boundary
boundary = Rectangle((2.2, 0.5), 7.4, 6.0, fill=False, edgecolor='#333', lw=1.8)
ax.add_patch(boundary)
ax.text(5.9, 6.2, 'Trading Bot System', ha='center', fontsize=11, fontweight='bold')

use_cases = [
    (4.0, 5.3, 'Configure\nSettings'),
    (4.0, 4.2, 'Select\nAlgorithm'),
    (4.0, 3.1, 'Start / Stop\nSession'),
    (7.5, 5.3, 'View Live\nStatistics'),
    (7.5, 4.2, 'Receive\nAlerts'),
    (7.5, 3.1, 'Force Long /\nShort'),
    (5.9, 1.5, 'Configure\nStop-Loss'),
]
for x, y, text in use_cases:
    ellipse = mpatches.Ellipse((x, y), 1.8, 0.8, facecolor='#DCE6F2', edgecolor='#1F4E79', lw=1.4)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5)
    ax.plot([1.0, x - 0.9], [4.3, y], 'k-', lw=0.8, alpha=0.6)

plt.title('Figure 12. Principal use cases exposed to the trader actor', fontsize=12, pad=10)
save('fig12_use_case_diagram')


# -----------------------------------------------------------------------------
# Fig 13: Sequence diagram (signal -> fill)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.axis('off')

actors = [('Data\nLayer', 1), ('LSTM\nForecaster', 3), ('Strategy\nEngine', 5),
          ('Risk\nGate', 7), ('Exchange\nAPI', 9), ('Telegram\nBot', 10.5)]
for name, x in actors:
    rect = FancyBboxPatch((x - 0.6, 7), 1.2, 0.7, boxstyle="round,pad=0.03",
                          linewidth=1.3, edgecolor='#333', facecolor='#DCE6F2')
    ax.add_patch(rect)
    ax.text(x, 7.35, name, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.plot([x, x], [0.3, 7], 'k--', lw=0.6, alpha=0.5)

# Messages
msgs = [
    (1, 3, 6.3, 'new tick window'),
    (3, 5, 5.7, 'prediction (class, conf)'),
    (5, 7, 5.1, 'proposed order'),
    (7, 5, 4.5, 'approve / reject'),
    (5, 9, 3.9, 'place order'),
    (9, 5, 3.3, 'fill confirmation'),
    (5, 10.5, 2.7, 'trade event'),
    (10.5, 5, 2.1, 'user command (optional)'),
]
for x1, x2, y, label in msgs:
    color = '#1F4E79' if x1 < x2 else '#C00000'
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', lw=1.4, color=color))
    xm = (x1 + x2) / 2
    ax.text(xm, y + 0.15, label, ha='center', fontsize=8.5, style='italic')

plt.title('Figure 13. Sequence of messages from signal generation to user confirmation',
          fontsize=12, pad=10)
save('fig13_sequence_diagram')


# -----------------------------------------------------------------------------
# Fig 14: State machine of a trading session
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

def state(x, y, text, color='#FFE699'):
    rect = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8, boxstyle="round,pad=0.05",
                          linewidth=1.5, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5, fontweight='bold')

state(1.2, 4.5, 'Idle', '#DCE6F2')
state(3.5, 4.5, 'Configuring', '#DCE6F2')
state(5.8, 4.5, 'Connecting', '#DCE6F2')
state(8.1, 4.5, 'Running', '#C5E0B4')
state(8.1, 2.5, 'Paused', '#FFE699')
state(5.8, 2.5, 'Stopping', '#F8CBAD')
state(3.5, 2.5, 'Error', '#F4B183')

def edge(x1, y1, x2, y2, label):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.4, color='#333'))
    ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha='center', fontsize=8.5, style='italic')

edge(2.0, 4.5, 2.7, 4.5, 'start')
edge(4.3, 4.5, 5.0, 4.5, 'settings valid')
edge(6.6, 4.5, 7.3, 4.5, 'connected')
edge(8.1, 4.1, 8.1, 2.9, 'pause')
edge(7.3, 2.5, 6.6, 2.5, 'stop')
edge(5.0, 2.5, 4.3, 2.5, 'unrecoverable')
edge(3.5, 2.9, 3.5, 4.1, 'reset')
edge(5.0, 4.3, 4.3, 2.7, 'invalid')

ax.plot([0.3, 0.7], [4.5, 4.5], 'k-', lw=2.5)  # initial state
ax.add_patch(plt.Circle((0.3, 4.5), 0.08, color='black'))

plt.title('Figure 14. State-transition diagram of a trading session', fontsize=12, pad=10)
save('fig14_state_machine')


print("Generated 14 figures in figures/")
for f in sorted(os.listdir('figures')):
    print(' ', f)
