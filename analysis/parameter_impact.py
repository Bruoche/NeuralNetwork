import os, sys, glob, json
import consts

import pandas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.monitor import MonitorModel

# For a set of models (same glob filter as the other analysis scripts), read the
# hyperparameters out of each params.json and report, per parameter, how the
# models did on average (best val_loss) grouped by that parameter's value.
#   python analysis/parameter_impact.py <prefix> [metric=val_loss]

METRIC = sys.argv[2] if len(sys.argv) > 2 else "val_loss"
prefix = sys.argv[1] if len(sys.argv) > 1 else ""


def extract_params(params):
	"""Flatten a params.json into a {parameter: value} dict. Augmentation knobs
	are normalized so 'off' is a real value (0 / (1,1)) that groups next to 'on'."""
	out = {"learning_rate": params.get("learning_rate")}
	shape = params.get("shape", {})
	layers = [shape[k] for k in shape if k.startswith("layer_")]

	dense = [l["size"] for l in layers if l.get("type") == "Dense"]
	out["hidden_layers"] = tuple(dense[:-1])  # head config (drops the class layer)

	for layer in layers:
		t = layer.get("type", "")
		if "base_model" in layer:                      # the frozen/fine-tuned backbone
			out["base_model"] = layer.get("base_model", t)
			out["fine_tune_blocks"] = layer.get("fine_tune_blocks", 0)
		elif t == "Dropout":
			out["dropout"] = layer.get("rate")
		elif t == "RandomRotation":
			deg = layer.get("degrees", [0, 0])
			out["rotation"] = round(abs(deg[-1]) / 360, 3)
		elif t == "RandomAffine":
			out["translation"] = tuple(layer.get("translate", [0, 0]))
			out["scale"] = tuple(layer.get("scale", [1, 1]))
		elif t in ("RandomHorizontalFlip", "RandomVerticalFlip"):
			out["h_flip" if "Horizontal" in t else "v_flip"] = layer.get("p", 0)
		elif t.startswith("Random") or t in ("ColorJitter", "GaussianBlur"):
			out[t] = "on"                              # generic fallback for other augments
	return out


rows = []
for path in sorted(glob.glob(f"{MonitorModel.MONITORING_DIR}/*/{prefix}*/metrics.csv")):
	metrics = pandas.read_csv(path)
	if METRIC not in metrics.columns or len(metrics) == 0:
		continue
	params_path = os.path.join(os.path.dirname(path), "params.json")
	params = json.load(open(params_path)) if os.path.exists(params_path) else {}
	row = extract_params(params)
	row["best"] = metrics[METRIC].min()
	rows.append(row)

if not rows:
	sys.exit(f"No usable models found for \"{prefix}\".")

df = pandas.DataFrame(rows)
# stringify unhashable / tuple values so they can be grouped and printed
for col in df.columns:
	if col != "best":
		df[col] = df[col].apply(lambda v: "absent" if pandas.isna(v) else (str(v) if isinstance(v, (tuple, list)) else v))

print(f"{len(df)} models matched \"{prefix}*\" | metric = best {METRIC} (lower = better)\n")

fixed = []
for col in [c for c in df.columns if c != "best"]:
	if df[col].nunique(dropna=False) <= 1:
		fixed.append(f"{col}={df[col].iloc[0]}")
		continue
	stats = (df.groupby(col)["best"]
			 .agg(mean="mean", best="min", n="count")
			 .sort_values("mean"))
	print(f"== {col} ==")
	for value, r in stats.iterrows():
		print(f"   {str(value):>14} : mean {r['mean']:.4f}   best {r['best']:.4f}   n {int(r['n'])}")
	print()

if fixed:
	print("fixed across this set:", ", ".join(fixed))
