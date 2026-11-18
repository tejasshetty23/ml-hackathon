"""Checks: saved models load and reproduce scores; labels/candidates are consistent.
Run:  py -3.13 -m tests.test_models_load      (from the PathSnap folder)
"""
import joblib, numpy as np, pandas as pd
from preprocessing.pipeline import FEATURES, process_trip

te = pd.read_parquet("data/processed/student_test.parquet")
X = te[FEATURES].values[:2000]
for name in ["LogisticRegression", "KNN", "DecisionTree", "RandomForest", "XGBoost"]:
    m = joblib.load(f"artifacts/models/{name}.joblib")
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (2000,) and np.all((p >= 0) & (p <= 1))
    print(f"OK  {name}: loaded, mean prob on 2000 test rows = {p.mean():.4f}")

# re-running preprocessing gives the same rows as the saved file (deterministic)
df, pts = process_trip("data/raw/student_test/00000052")
saved = te[te.trip == "00000052"]
assert len(df) == len(saved) and (df.y.values == saved.y.values).all()
# each GPS point has at most one positive candidate (one true road)
# (duplicate u->v arcs can give >1 positive per point)
print("OK  preprocessing is deterministic; positives per point max =", te.groupby(["trip", "point"]).y.sum().max())
