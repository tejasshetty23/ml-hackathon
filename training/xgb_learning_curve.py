"""Real training/validation log-loss per boosting round for the final XGBoost model.
(XGBoost is the only one of our models with an epoch-like training history; no curves are drawn for the others.)

Run:  py -3.13 -m training.xgb_learning_curve      (from the PathSnap folder, after training.tune_bayes)
Split: 64 training trips / 16 validation trips (GroupShuffleSplit by trip, seed 42) - training partition only.
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from preprocessing.pipeline import FEATURES
from training.train_models import SEED

cfg = json.load(open("artifacts/final_config.json"))
params = json.load(open("results/metrics/bayes_best.json"))["best_params"] if cfg["model_name"] == "XGBoost_tuned" else \
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
tr = pd.read_parquet("data/processed/train.parquet")
a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, groups=tr.trip))
X, y = tr[FEATURES].values, tr.y.values
m = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss", **params)
m.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], verbose=False)
h = m.evals_result()
curve = pd.DataFrame({"round": range(1, len(h["validation_0"]["logloss"]) + 1),
                      "train_logloss": h["validation_0"]["logloss"], "val_logloss": h["validation_1"]["logloss"]})
curve.to_csv("results/metrics/xgb_learning_curve.csv", index=False)
plt.figure(figsize=(7, 3.6))
plt.plot(curve["round"], curve.train_logloss, label="training trips (64)")
plt.plot(curve["round"], curve.val_logloss, label="validation trips (16)")
best = curve.val_logloss.idxmin()
plt.axvline(curve["round"][best], color="grey", ls="--", lw=0.8)
plt.title(f"{cfg['model_name']}: log loss per boosting round (min val at round {curve['round'][best]})")
plt.xlabel("boosting round"); plt.ylabel("log loss"); plt.legend(); plt.tight_layout()
plt.savefig("results/plots/xgb_learning_curve.png", dpi=130)
print(curve.iloc[[0, best, len(curve) - 1]])
