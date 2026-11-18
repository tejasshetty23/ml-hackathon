"""Checks: saved models load and predict; preprocessing is deterministic; inference does not need the route.
Run:  py -3.13 -m tests.test_models_load      (from the PathSnap folder)
"""
import json
import joblib, numpy as np, pandas as pd
from preprocessing.pipeline import FEATURES, process_trip, read_trip_files, clean_track, candidates_and_features
from models.hmm import RoadGraph, prepare_trip, viterbi

te = pd.read_parquet("data/processed/student_test.parquet")
X = te[FEATURES].values[:2000]
for name in ["LogisticRegression", "KNN", "DecisionTree", "RandomForest", "XGBoost", "XGBoost_tuned"]:
    m = joblib.load(f"artifacts/models/{name}.joblib")
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (2000,) and np.all((p >= 0) & (p <= 1))
    print(f"OK  {name}: loaded, mean prob on 2000 student-test rows = {p.mean():.4f}")

# preprocessing is deterministic
trip = "data/raw/student_test/00000052"
df, pts, _ = process_trip(trip)
saved = te[te.trip == "00000052"]
assert len(df) == len(saved) and (df.y.values == saved.y.values).all()
print("OK  preprocessing is deterministic")

# inference path (what the API runs) works without the .route file and gives identical features
tid, track, nodes, arcs, _ = read_trip_files(trip)
track, _ = clean_track(track)
inf, geo = candidates_and_features(tid, track, nodes, arcs)
assert np.allclose(inf[FEATURES].values, saved[FEATURES].values) and "y" not in inf.columns
cfg = json.load(open("artifacts/final_config.json"))
inf["p"] = joblib.load(cfg["model_file"]).predict_proba(inf[FEATURES].values)[:, 1]
g = RoadGraph(arcs.u.values, arcs.v.values, geo["arc_len"], len(nodes))
pred = viterbi(prepare_trip(inf, geo["px"], geo["py"], geo["t"], g, cfg["hmm"]["gap_s"]),
               cfg["hmm"]["beta"], cfg["hmm"]["w_emission"])
assert len(pred) == (geo["n_cands"] > 0).sum()
print(f"OK  route-free inference + HMM: {len(pred)} points decoded with {cfg['model_name']}")
