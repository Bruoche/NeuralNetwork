import os, sys, glob, re
import consts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.monitor import MonitorModel
from pathlib import Path

METRIC_SHOWN = "val_loss"

prefix = sys.argv[1] if len(sys.argv) > 1 else ""

paths = sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/*/{prefix}*/metrics.csv"))
if not paths:
	sys.exit(f"No usable metrics found for \"{prefix}\".")

all_metrics = pandas.concat([pandas.read_csv(path) for path in paths], ignore_index=True)
bests = all_metrics.groupby("model")[METRIC_SHOWN].min().sort_values()

PLOT_WIDTH, PLOT_HEIGHT, LEGEND_FONT = 9, 5, 7
MAX_LEGEND_WIDTH = 9  # inches, so the window still fits on a screen

# the part of the name shared by every model is already in the title: drop it
# from the labels so long names stay readable in the legend
common = os.path.commonprefix(list(bests.index))
common = common[:common.rfind("_") + 1]
labels = {model: f"{bests[model]:.4f}: {model[len(common):] or model}" for model in bests.index}

# size the legend from what actually fits next to the plot: a full label per
# column, as many rows as the figure is tall. Past that only the best models
# get a legend entry (the others are still plotted).
column_width = 0.6 + max(len(label) for label in labels.values()) * LEGEND_FONT * 0.6 / 72
rows = max(1, int(PLOT_HEIGHT / (LEGEND_FONT * 1.6 / 72)))
max_columns = max(1, int(MAX_LEGEND_WIDTH / column_width))
labelled = list(bests.index[:rows * max_columns])
ncol = 1 + (len(labelled) - 1) // rows
legend_width = ncol * column_width

fig, ax = plt.subplots(figsize=(PLOT_WIDTH + legend_width, PLOT_HEIGHT))
for model in bests.index:
	group = all_metrics[all_metrics.model == model]
	ax.plot(group["epoch"], group[METRIC_SHOWN],
			label=labels[model] if model in labelled else "_nolegend_")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (test)")
hidden = len(bests) - len(labelled)
ax.set_title(f"{prefix}* - Fitness Comparisons"
			 + (f" (legend: best {len(labelled)} of {len(bests)})" if hidden else ""))
ax.legend(
	fontsize=LEGEND_FONT,
	loc="upper left",
	bbox_to_anchor=(1.01, 1.0),
	ncol=ncol
)
ax.grid(alpha=.3)
ax.set_yscale("log")
best = bests.min()
if len(bests) >= 4:
	q1, q3 = bests.quantile([0.25, 0.75])
	worst_shown = bests[bests <= q3 + 1.5 * (q3 - q1)].max()
else:
	worst_shown = bests.max()
ax.set_ylim(worst_shown * 1.3, best * 0.95)

# tight_layout ignores artists placed outside the axes, so reserve the legend
# strip by hand instead: everything right of `right` is legend space
fig.subplots_adjust(left=0.9 / fig.get_figwidth(), right=1 - legend_width / fig.get_figwidth())
Path(consts.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
normalized_prefix = re.sub(r'[^\w.-]', "_", prefix).strip("_").lstrip("_")
if not normalized_prefix:
	normalized_prefix = "all"
plt.savefig(f"{consts.GRAPHS_DIR}/comparison_{normalized_prefix}_loss.png", dpi=150, bbox_inches="tight")
plt.show()