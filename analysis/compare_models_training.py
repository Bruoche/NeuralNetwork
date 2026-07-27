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

paths = sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/{prefix}*/metrics.csv"))
if not paths:
	sys.exit(f"No usable metrics found for \"{prefix}\".")

all_metrics = pandas.concat([pandas.read_csv(path) for path in paths], ignore_index=True)
bests = all_metrics.groupby("model")[METRIC_SHOWN].min().sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
for model in bests.index:
	group = all_metrics[all_metrics.model == model]
	ax.plot(group["epoch"], group[METRIC_SHOWN], label=f"{bests[model]:.4f}: {model}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (test)")
ax.set_title(f"{prefix}* - Fitness Comparisons")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
ax.set_yscale("log")
ax.relim()
ax.autoscale_view()
min, max = ax.get_ylim()
ax.set_ylim(max, min)

plt.tight_layout()
Path(consts.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
normalized_prefix = re.sub(r'[^\w.-]', "_", prefix).strip("_").lstrip("_")
if not normalized_prefix:
	normalized_prefix = "all"
plt.savefig(f"{consts.GRAPHS_DIR}/comparison_{normalized_prefix}_loss.png", dpi=150)
plt.show()