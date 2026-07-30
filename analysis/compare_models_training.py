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
# optional 2nd arg: only plot the N best models (0 / absent = plot all)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

paths = sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/*/{prefix}*/metrics.csv"))
if not paths:
	sys.exit(f"No usable metrics found for \"{prefix}\".")

all_metrics = pandas.concat([pandas.read_csv(path) for path in paths], ignore_index=True)
bests = all_metrics.groupby("model")[METRIC_SHOWN].min().sort_values()
total_models = len(bests)
if limit > 0:
	bests = bests.head(limit)

PLOT_WIDTH, PLOT_HEIGHT, LEGEND_FONT = 9, 5, 7
MAX_LEGEND_WIDTH = 9

# strip the CLI prefix (already shown in the title) so labels stay readable and
# consistent regardless of how many models are plotted; keep the rest of the name
labels = {model: f"{bests[model]:.4f}: {model}"
		  for model in bests.index}

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
title = f"{prefix}* - Fitness Comparisons"
if limit > 0 and len(bests) < total_models:
	title += f" (best {len(bests)} of {total_models})"
if hidden:
	title += f" (legend: {len(labelled)} shown)"
ax.set_title(title)
ax.legend(
	fontsize=LEGEND_FONT,
	loc="upper left",
	bbox_to_anchor=(1.01, 1.0),
	ncol=ncol
)
ax.grid(alpha=.3)
ax.set_yscale("log")
# y-limits span the full curves of the plotted models: top = best value seen,
# bottom = the biggest value seen (the worst plotted curve's peak), so each
# shown model is visible start-to-finish. Axis is inverted (lower loss = up).
# Narrow the plotted set with the 2nd CLI arg to keep bad runs from stretching it.
shown = all_metrics[all_metrics.model.isin(bests.index)][METRIC_SHOWN]
ax.set_ylim(shown.max() * 1.05, shown.min() * 0.95)

fig.subplots_adjust(left=0.9 / fig.get_figwidth(), right=1 - legend_width / fig.get_figwidth())
Path(consts.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
normalized_prefix = re.sub(r'[^\w.-]', "_", prefix).strip("_").lstrip("_")
if not normalized_prefix:
	normalized_prefix = "all"
plt.savefig(f"{consts.GRAPHS_DIR}/comparison_{normalized_prefix}_loss.png", dpi=150, bbox_inches="tight")
plt.show()