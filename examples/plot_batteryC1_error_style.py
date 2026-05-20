import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / 'C1-500.csv'
OUTPUT_PATH = SCRIPT_DIR / 'C1_capacity_error.png'


SP  = 500
EOL = 0.77


FIG_WIDTH  = 7.0
FIG_HEIGHT = 6.4
DPI        = 300


XLIM = (0, 1000)
YLIM = (0.0, 1.4)


ZOOM_XLIM  = (630, 685)
ZOOM_YLIM  = (0.72, 0.82)
INSET_BBOX = [0.14, 0.13, 0.46, 0.38]


PANEL_LABEL = '(c)'


FONT_FAMILY = 'Times New Roman'


available = {f.name for f in font_manager.fontManager.ttflist}
if FONT_FAMILY not in available:
    print(f"[Warning] '{FONT_FAMILY}' is not installed. Falling back to serif.")
    FONT_FAMILY = 'serif'

mpl.rcParams['font.family']        = FONT_FAMILY
mpl.rcParams['mathtext.fontset']   = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth']     = 0.95
mpl.rcParams['xtick.major.width']  = 0.90
mpl.rcParams['ytick.major.width']  = 0.90
mpl.rcParams['savefig.facecolor']  = 'white'


MODEL_STYLE = {
    'LSTM':           ('#E69F00', 'o', 3.6),
    'TCN':            ('#0072B2', 's', 3.6),
    'DLinear':        ('#009E73', 'D', 3.6),
    'PatchTST':       ('#CC79A7', '^', 3.8),
    'iTransformer':   ('#56B4E9', 'v', 3.8),
    'NBD-Net (Ours)': ('#D55E00', 'X', 4.6),
}
REAL_COLOR      = '#4D4D4D'
THRESHOLD_COLOR = '#1A1A1A'
SP_COLOR        = '#7A1FA2'
GRID_COLOR      = '#D9D9D9'


df = pd.read_csv(CSV_PATH)
cycle = df['Cycle'].to_numpy()
real = df['Capacity(Ah)'].to_numpy()


model_names = [name for name in MODEL_STYLE if name in df.columns]


fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[3.55, 1.25],
    hspace=0.10
)
ax = fig.add_subplot(gs[0])
ax_err = fig.add_subplot(gs[1], sharex=ax)


ax.plot(
    cycle, real,
    color=REAL_COLOR,
    linewidth=1.8,
    label='Real data',
    zorder=2
)

for name in model_names:
    color, marker, ms = MODEL_STYLE[name]
    y = df[name].to_numpy()
    mask = ~np.isnan(y)
    ax.plot(
        cycle[mask], y[mask],
        color=color,
        marker=marker,
        markersize=ms,
        markevery=20,
        linewidth=1.15,
        label=name,
        zorder=3,
        alpha=0.95
    )


ax.axhline(
    y=EOL,
    color=THRESHOLD_COLOR,
    linestyle='--',
    linewidth=1.05,
    label='Failure threshold',
    zorder=1
)
ax.axvline(
    x=SP,
    color=SP_COLOR,
    linestyle='-',
    linewidth=1.45,
    zorder=1
)

ax.set_ylabel('Capacity (Ah)', fontsize=14)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.tick_params(labelsize=12, direction='in', top=True, right=True)
ax.grid(True, which='major', linestyle='--', linewidth=0.55, color=GRID_COLOR, alpha=0.65)


plt.setp(ax.get_xticklabels(), visible=False)


handles, labels = ax.get_legend_handles_labels()
order = ['Failure threshold', 'Real data',
         'LSTM', 'TCN', 'DLinear', 'PatchTST', 'iTransformer', 'NBD-Net (Ours)']
idx = [labels.index(l) for l in order if l in labels]
ax.legend(
    [handles[i] for i in idx],
    [labels[i] for i in idx],
    loc='upper right',
    ncol=3,
    fontsize=8.8,
    frameon=True,
    fancybox=False,
    edgecolor='#BFBFBF',
    facecolor='white',
    framealpha=0.92,
    handletextpad=0.35,
    columnspacing=0.75,
    labelspacing=0.28,
    borderpad=0.35
)


axins = ax.inset_axes(INSET_BBOX)
axins.plot(cycle, real, color=REAL_COLOR, linewidth=1.55)
for name in model_names:
    color, marker, ms = MODEL_STYLE[name]
    y = df[name].to_numpy()
    mask = ~np.isnan(y)
    axins.plot(
        cycle[mask], y[mask],
        color=color,
        marker=marker,
        markersize=max(ms - 0.4, 2.8),
        markevery=3,
        linewidth=1.0,
        alpha=0.95
    )
axins.axhline(y=EOL, color=THRESHOLD_COLOR, linestyle='--', linewidth=0.95)
axins.set_xlim(*ZOOM_XLIM)
axins.set_ylim(*ZOOM_YLIM)
axins.tick_params(labelsize=8.8, direction='in', top=True, right=True)
axins.grid(True, linestyle='--', linewidth=0.45, color=GRID_COLOR, alpha=0.60)


ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.75, linewidth=0.8)


ax.text(
    0.965, 0.055, PANEL_LABEL,
    transform=ax.transAxes,
    fontsize=16,
    fontweight='bold',
    va='bottom',
    ha='right'
)


for name in model_names:
    color, marker, ms = MODEL_STYLE[name]
    y = df[name].to_numpy()
    error = y - real
    mask = ~np.isnan(error)

    ax_err.plot(
        cycle[mask], error[mask],
        color=color,
        linewidth=0.95,
        alpha=0.82,
        zorder=2
    )
    ax_err.scatter(
        cycle[mask], error[mask],
        color=color,
        marker=marker,
        s=(ms + 1.2) ** 2,
        alpha=0.72,
        edgecolors='none',
        zorder=3
    )

ax_err.axhline(y=0, color='#262626', linestyle='--', linewidth=0.95, zorder=1)
ax_err.axvline(x=SP, color=SP_COLOR, linestyle='-', linewidth=1.2, alpha=0.85, zorder=1)
ax_err.set_xlabel('Cycle', fontsize=14)
ax_err.set_ylabel('Error (Ah)', fontsize=13)
ax_err.tick_params(labelsize=11.5, direction='in', top=True, right=True)
ax_err.grid(True, which='major', linestyle='--', linewidth=0.50, color=GRID_COLOR, alpha=0.65)


all_errors = []
for name in model_names:
    err = df[name].to_numpy() - real
    err = err[~np.isnan(err)]
    if err.size:
        all_errors.append(err)
if all_errors:
    all_errors = np.concatenate(all_errors)
    lo, hi = np.nanpercentile(all_errors, [1, 99])
    pad = max((hi - lo) * 0.18, 1e-4)
    ax_err.set_ylim(lo - pad, hi + pad)


fig.align_ylabels([ax, ax_err])
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
print(f"Saved: {OUTPUT_PATH}")
plt.show()
