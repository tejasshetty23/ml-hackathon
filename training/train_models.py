"""Train + evaluate the 5 models.

Run:  py -3.13 -m training.train_models      (from the PathSnap folder)
- 5-fold GroupKFold on the 80 training trips (a trip never appears in both train and validation fold)
- refit on all 80 training trips, evaluate on the 10 student-testing trips
- saves models to artifacts/models/, metrics to results/metrics/
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from xgboost import XGBClassifier

from preprocessing.pipeline import FEATURES

SEED = 42
KNN_MAX_TRAIN = 50_000  # KNN stores all training rows; subsample to keep prediction time reasonable

MODELS = {
    "LogisticRegression": ("Linear", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
    "KNN": ("Instance-based", lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="brute", n_jobs=-1))),
    "DecisionTree": ("Tree", lambda: DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=50, random_state=SEED)),
    "RandomForest": ("Bagging ensemble", lambda: RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=20, max_samples=0.3, n_jobs=-1, random_state=SEED)),
    "XGBoost": ("Boosting", lambda: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss")),
}


def fit(name, model, X, y):
    if name == "KNN" and len(X) > KNN_MAX_TRAIN:
        idx = np.random.default_rng(SEED).choice(len(X), KNN_MAX_TRAIN, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.time()
    model.fit(X, y)
    return time.time() - t0


def route_continuity(pred_rows):
    """Share of consecutive GPS points whose predicted roads are the same arc or directly connected."""
    ok, total = 0, 0
    for _, g in pred_rows.groupby("trip"):
        g = g.sort_values("point")
        same = g.arc.values[1:] == g.arc.values[:-1]
        conn = g.v.values[:-1] == g.u.values[1:]
        ok += (same | conn).sum(); total += len(g) - 1
    return ok / max(total, 1)


def evaluate(df, points, prob):
    """Candidate-level classification metrics + observation-level Top-1 accuracy + continuity."""
    y = df.y.values
    pred = (prob >= 0.5).astype(int)
    d = df[["trip", "point", "arc", "u", "v", "y"]].copy(); d["p"] = prob
    top1 = d.loc[d.groupby(["trip", "point"]).p.idxmax()]
    n_points = len(points)  # GPS points with no candidate at all count as wrong
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "top1_accuracy": top1.y.sum() / n_points,
        "top1_accuracy_covered": top1.y.sum() / points.label_in_cands.sum(),
        "candidate_miss_rate": 1 - points.label_in_cands.mean(),
        "route_continuity": route_continuity(top1),
    }, confusion_matrix(y, pred, labels=[0, 1]), top1


def main():
    os.makedirs("artifacts/models", exist_ok=True); os.makedirs("results/metrics", exist_ok=True)
    tr = pd.read_parquet("data/processed/train.parquet")
    tr_pts = pd.read_parquet("data/processed/train_points.parquet")
    te = pd.read_parquet("data/processed/student_test.parquet")
    te_pts = pd.read_parquet("data/processed/student_test_points.parquet")
    Xtr, ytr, groups = tr[FEATURES].values, tr.y.values, tr.trip.values
    Xte = te[FEATURES].values
    print(f"train rows={len(tr)} (pos rate {ytr.mean():.3f}), student-test rows={len(te)}", flush=True)

    cv_rows, test_rows, summary = [], [], []
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, groups))
    names = sys.argv[1:] or list(MODELS)  # optionally train only some models, e.g. "KNN"
    for name in names:
        family, make = MODELS[name]
        print(f"\n=== {name}", flush=True)
        oof = np.zeros(len(tr))  # out-of-fold probability for every training row
        for k, (a, b) in enumerate(folds):
            m = make()
            fit_s = fit(name, m, Xtr[a], ytr[a])
            t0 = time.time(); p = m.predict_proba(Xtr[b])[:, 1]; inf_s = time.time() - t0
            oof[b] = p
            val = tr.iloc[b]
            met, _, _ = evaluate(val, tr_pts[tr_pts.trip.isin(val.trip.unique())], p)
            cv_rows.append({"model": name, "fold": k, "train_time_s": fit_s, "inference_time_s": inf_s, **met})
            print(f"  fold {k}: macroF1={met['macro_f1']:.4f} top1={met['top1_accuracy']:.4f} ({fit_s:.1f}s fit)", flush=True)

        m = make()
        fit_s = fit(name, m, Xtr, ytr)
        t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t0
        met, cm, top1 = evaluate(te, te_pts, p)
        # overfitting check on the SAME 20 random training trips: score of the refit model (trips seen in training)
        # vs out-of-fold score (same trips, predicted by the fold model that did not see them)
        sub_trips = np.random.default_rng(SEED).choice(tr.trip.unique(), 20, replace=False)
        sel = tr.trip.isin(sub_trips).values
        tr_sub, pts_sub = tr[sel], tr_pts[tr_pts.trip.isin(sub_trips)]
        tr_met, _, _ = evaluate(tr_sub, pts_sub, m.predict_proba(tr_sub[FEATURES].values)[:, 1])
        oof_met, _, _ = evaluate(tr_sub, pts_sub, oof[sel])
        np.save(f"data/processed/oof_{name}.npy", oof)  # out-of-fold probabilities (used to tune the HMM)
        joblib.dump(m, f"artifacts/models/{name}.joblib")
        np.save(f"results/metrics/confusion_{name}.npy", cm)
        top1[["trip", "point", "arc", "p"]].to_csv(f"results/metrics/student_test_predictions_{name}.csv", index=False)
        test_rows.append({"model": name, "train_time_s": fit_s, "inference_time_s": inf_s,
                          "inference_ms_per_point": 1000 * inf_s / len(te_pts), **met})
        cv = pd.DataFrame([r for r in cv_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": json.dumps(get_params(m)),
            "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1_mean": cv.top1_accuracy.mean(), "cv_log_loss_mean": cv.log_loss.mean(),
            "train_macro_f1": tr_met["macro_f1"], "train_top1": tr_met["top1_accuracy"],
            "oof_macro_f1_same_trips": oof_met["macro_f1"], "oof_top1_same_trips": oof_met["top1_accuracy"],
            "test_accuracy": met["accuracy"], "test_precision": met["precision"], "test_recall": met["recall"],
            "test_macro_f1": met["macro_f1"], "test_top1": met["top1_accuracy"], "test_log_loss": met["log_loss"],
            "test_route_continuity": met["route_continuity"], "candidate_miss_rate": met["candidate_miss_rate"],
            "train_time_s": fit_s, "inference_ms_per_point": 1000 * inf_s / len(te_pts)})
        print(f"  student-test: macroF1={met['macro_f1']:.4f} top1={met['top1_accuracy']:.4f} "
              f"logloss={met['log_loss']:.4f} continuity={met['route_continuity']:.4f}", flush=True)

        # one file set per model, then merged into the three combined CSVs (in fixed model order)
        os.makedirs("results/metrics/per_model", exist_ok=True)
        for tag, rows in [("cv_results", cv_rows), ("student_test_results", test_rows), ("model_comparison", summary)]:
            pd.DataFrame([r for r in rows if r["model"] == name]).to_csv(
                f"results/metrics/per_model/{tag}_{name}.csv", index=False)
            parts = [f"results/metrics/per_model/{tag}_{n}.csv" for n in MODELS]
            pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)]).to_csv(
                f"results/metrics/{tag}.csv", index=False)

    # reference-label continuity = best achievable value of the metric
    ref = te.loc[te.y == 1].drop_duplicates(["trip", "point"])
    print(f"\nReference-route continuity on student-test (ceiling): {route_continuity(ref):.4f}")


def get_params(m):
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    keys = {"LogisticRegression": ["C", "max_iter"], "KNeighborsClassifier": ["n_neighbors", "weights"],
            "DecisionTreeClassifier": ["max_depth", "min_samples_leaf"],
            "RandomForestClassifier": ["n_estimators", "max_depth", "min_samples_leaf", "max_samples"],
            "XGBClassifier": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]}
    p = est.get_params()
    return {k: p.get(k) for k in keys.get(type(est).__name__, [])}


if __name__ == "__main__":
    main()
