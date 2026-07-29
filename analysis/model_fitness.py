import os, sys, glob
import consts

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.monitor import MonitorModel
from pathlib import Path

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
paths = sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/*/{prefix}*/metrics.csv"))
if not paths:
	sys.exit(f"No model found for \"{prefix}\".")
all_metrics = pandas.read_csv(paths[-1])
model = all_metrics["model"][0]

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for i, metric in enumerate(["loss", "accuracy"]):
	ax = axes[i]
	test_metric = f"val_{metric}"
	ax.plot(all_metrics["epoch"], all_metrics[test_metric], label=f"{all_metrics[test_metric].min():.4f}: test")
	ax.plot(all_metrics["epoch"], all_metrics[metric], label=f"{all_metrics[metric].min():.4f}: train")
	ax.set_xlabel("Epoch")
	ax.set_ylabel(str.capitalize(metric))
	ax.set_title(f"{model} - {metric}")
	ax.legend(fontsize=8)
	ax.grid(alpha=.3)
	ax.set_yscale("log")

plt.tight_layout()
Path(consts.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
plt.savefig(f"{consts.GRAPHS_DIR}/metrics_{model}.png", dpi=150)
plt.show()
