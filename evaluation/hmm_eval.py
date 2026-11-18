"""Tune the HMM decoder on training trips and compare classifier-only vs classifier+HMM on student-testing trips.

Run:  py -3.13 -m evaluation.hmm_eval      (from the PathSnap folder, after training.tune_bayes)
- final classifier = tuned XGBoost if its 5-fold CV Top-1 beats the untuned one, else untuned XGBoost
- decoder parameters (beta, emission weight) are chosen on 20 training trips using OUT-OF-FOLD probabilities
  (so the classifier never saw those trips), never on student-testing or evaluator data
- writes artifacts/final_config.json (used by the API) and results/metrics/hmm_*.csv
"""
import os, json, time
import numpy as np
import pandas as pd
import joblib

from preprocessing.pipeline import RAW, FEATURES, load_trip, to_metres
from models.hmm import RoadGraph, prepare_trip, viterbi
from training.train_models import route_continuity, SEED

BETAS = [3, 5, 10, 20, 40]
W_ES = [0.5, 1.0, 2.0]
GAP_S = 60.0


def trip_geometry(trip_dir):
    tid, track, nodes, arcs, route = load_trip(trip_dir)
    lat0 = track.lat.mean()
    nx, ny = to_metres(nodes.lon.values, nodes.lat.values, lat0)
    px, py = to_metres(track.lon.values, track.lat.values, lat0)
    u, v = arcs.u.values, arcs.v.values
    graph = RoadGraph(u, v, np.hypot(nx[v] - nx[u], ny[v] - ny[u]), len(nodes))
    return px, py, track.t.values.astype(float), graph, route, arcs


def predictions_table(trip, pred, arcs):
    pts = np.array(sorted(pred)); a = np.array([pred[p] for p in pts], dtype=int)
    return pd.DataFrame({"trip": trip, "point": pts, "arc": a, "u": arcs.u.values[a], "v": arcs.v.values[a]})


def arc_set_f1(pred_arcs, route):
    P, R = set(pred_arcs), set(route)
    if not P:
        return 0.0, 0.0, 0.0
    prec, rec = len(P & R) / len(P), len(P & R) / len(R)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def score(pred_df, pts):
    """Top-1 over all labelled GPS points (points without a prediction count as wrong)."""
    m = pts.merge(pred_df[["trip", "point", "arc"]], on=["trip", "point"], how="left")
    return float((m.arc == m.label_arc).sum() / len(m))


def main():
    tvu = pd.read_csv("results/metrics/tuned_vs_untuned.csv").set_index("model")
    use_tuned = tvu.loc["XGBoost (tuned)", "cv_top1"] >= tvu.loc["XGBoost (untuned)", "cv_top1"]
    model_name = "XGBoost_tuned" if use_tuned else "XGBoost"
    model = joblib.load(f"artifacts/models/{model_name}.joblib")
    print("final classifier:", model_name)

    # ---- 1. decoder tuning on 30 training trips with out-of-fold probabilities ----
    tr = pd.read_parquet("data/processed/train.parquet")
    tr["p"] = np.load(f"data/processed/oof_{model_name}.npy")
    tr_pts = pd.read_parquet("data/processed/train_points.parquet")
    trips = np.random.default_rng(SEED).choice(sorted(tr.trip.unique()), 20, replace=False)
    grid = {(b, w): [] for b in BETAS for w in W_ES}
    cls_only = []
    t0 = time.time()
    for trip in trips:
        px, py, t, graph, route, arcs = trip_geometry(f"{RAW}/train/{trip}")
        c = tr[tr.trip == trip]
        steps = prepare_trip(c, px, py, t, graph, gap_s=GAP_S)
        cls_only.append(c.loc[c.groupby("point").p.idxmax(), ["trip", "point", "arc"]])
        for (b, w) in grid:
            grid[(b, w)].append(predictions_table(trip, viterbi(steps, b, w), arcs))
    sel_pts = tr_pts[tr_pts.trip.isin(trips)]
    rows = [{"beta": b, "w_emission": w, "train_top1_hmm": score(pd.concat(v), sel_pts)} for (b, w), v in grid.items()]
    g = pd.DataFrame(rows).sort_values("train_top1_hmm", ascending=False)
    g["train_top1_classifier_only"] = score(pd.concat(cls_only), sel_pts)
    g.to_csv("results/metrics/hmm_grid.csv", index=False)
    best = g.iloc[0]
    print(g.head(5).to_string(index=False), f"\n(decoder tuning {time.time() - t0:.0f}s)")

    cfg = {"model_file": f"artifacts/models/{model_name}.joblib", "model_name": model_name,
           "hmm": {"beta": float(best.beta), "w_emission": float(best.w_emission), "gap_s": GAP_S},
           "features": FEATURES}
    json.dump(cfg, open("artifacts/final_config.json", "w"), indent=2)

    # ---- 2. student-testing comparison: classifier-only vs classifier + HMM ----
    te = pd.read_parquet("data/processed/student_test.parquet")
    te_pts = pd.read_parquet("data/processed/student_test_points.parquet")
    te["p"] = model.predict_proba(te[FEATURES].values)[:, 1]
    per_trip, all_cls, all_hmm = [], [], []
    for trip in sorted(te.trip.unique()):
        px, py, t, graph, route, arcs = trip_geometry(f"{RAW}/student_test/{trip}")
        c = te[te.trip == trip]
        t0 = time.time()
        hmm = predictions_table(trip, viterbi(prepare_trip(c, px, py, t, graph, GAP_S), best.beta, best.w_emission), arcs)
        dec_s = time.time() - t0
        top = c.loc[c.groupby("point").p.idxmax(), ["trip", "point", "arc", "u", "v"]]
        p_pts = te_pts[te_pts.trip == trip]
        row = {"trip": trip, "points": len(p_pts), "hmm_decode_s": dec_s}
        for tag, pred in [("cls", top), ("hmm", hmm)]:
            prec, rec, f1 = arc_set_f1(pred.arc.unique(), route)
            row.update({f"{tag}_top1": score(pred, p_pts), f"{tag}_continuity": route_continuity(pred),
                        f"{tag}_route_precision": prec, f"{tag}_route_recall": rec, f"{tag}_route_f1": f1})
        per_trip.append(row); all_cls.append(top); all_hmm.append(hmm)
        print(f"  {trip}: top1 {row['cls_top1']:.4f} -> {row['hmm_top1']:.4f}, continuity "
              f"{row['cls_continuity']:.4f} -> {row['hmm_continuity']:.4f}", flush=True)
    pt = pd.DataFrame(per_trip)
    pt.to_csv("results/metrics/hmm_per_trip.csv", index=False)
    cls, hmm = pd.concat(all_cls), pd.concat(all_hmm)
    hmm.to_csv("results/metrics/student_test_predictions_hmm.csv", index=False)
    cls.to_csv("results/metrics/student_test_predictions_final_classifier.csv", index=False)
    summary = pd.DataFrame([
        {"method": f"{model_name} only", "top1": score(cls, te_pts), "route_continuity": route_continuity(cls),
         "route_f1_mean": pt.cls_route_f1.mean(), "route_precision_mean": pt.cls_route_precision.mean(),
         "route_recall_mean": pt.cls_route_recall.mean()},
        {"method": f"{model_name} + HMM", "top1": score(hmm, te_pts), "route_continuity": route_continuity(hmm),
         "route_f1_mean": pt.hmm_route_f1.mean(), "route_precision_mean": pt.hmm_route_precision.mean(),
         "route_recall_mean": pt.hmm_route_recall.mean()}])
    summary.to_csv("results/metrics/hmm_comparison.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
