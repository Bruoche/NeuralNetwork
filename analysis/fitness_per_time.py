import os, sys, glob, json, re
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

rows = []
skipped = []
for path in paths:
	metrics = pandas.read_csv(path)
	model = metrics["model"][0]
	best_val_loss = metrics["val_loss"].min()

	params_path = os.path.join(os.path.dirname(path), "params.json")
	duration = None
	if os.path.exists(params_path):
		duration = json.load(open(params_path)).get("training_duration")
	if not duration:
		skipped.append(model)
		continue
	seconds = pandas.to_timedelta(duration).total_seconds()
	if seconds <= 0:
		skipped.append(model)
		continue

	rows.append({
		"model": model,
		"best_val_loss": best_val_loss,
		"seconds": seconds,
		"score": best_val_loss * seconds,
	})

if skipped:
	print(f"Skipped {len(skipped)} model(s) without training_duration: {', '.join(skipped)}")
if not rows:
	sys.exit("No model had a usable training_duration.")

data = pandas.DataFrame(rows).sort_values("score").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(max(9, 0.4 * len(data)), 5))
bars = ax.bar(data["model"], data["score"], color="tab:blue")
ax.bar_label(bars, labels=[f"{s:.2e}" for s in data["score"]], fontsize=6, rotation=90, padding=2)

ax.set_xlabel("Model")
ax.set_ylabel("val_loss * seconds  (lower = better)")
ax.set_title(f"{prefix}* - Loss x training-time cost")
ax.set_xticks(range(len(data)))
ax.set_xticklabels(data["model"], rotation=90, fontsize=7)
ax.grid(axis="y", alpha=.3)

plt.tight_layout()
Path(consts.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
normalized_prefix = re.sub(r'[^\w.-]', "_", prefix).strip("_").lstrip("_") or "all"
plt.savefig(f"{consts.GRAPHS_DIR}/fitness_per_time_{normalized_prefix}.png", dpi=150, bbox_inches="tight")
plt.show()
