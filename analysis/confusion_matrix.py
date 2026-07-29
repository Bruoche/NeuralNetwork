import os, sys, glob, json, pandas, pathlib
import matplotlib.pyplot as plt
import numpy as np
import consts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.monitor import MonitorModel

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
paths_metrics = sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/*/{prefix}*/confusion.csv"))
if not paths_metrics:
	sys.exit(f"No model found for \"{prefix}\".")
path = paths_metrics[-1]
confusion = pandas.read_csv(path)
model = confusion["model"][0]
matrix = confusion.pivot(index="wanted", columns="prediction", values="count")
classes = list(matrix.index)

percent = matrix.div(matrix.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"{confusion['model'][0]} - confusion (%)")

for i in range(len(classes)):
    for j in range(len(classes)):
        v = percent.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 50 else "black")

fig.colorbar(im, label="% of actual class")
plt.tight_layout()

plt.savefig(f"{consts.GRAPHS_DIR}/confusion_{model}.png", dpi=150)
plt.show()