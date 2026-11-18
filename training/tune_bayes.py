"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (XGBoost).

Run:  py -3.13 -m training.tune_bayes      (from the PathSnap folder, after training.train_models)
- objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold on the 80 training trips
  (trials use a fixed 25% sample of GPS points per trip to keep each trial ~30 s)
- best configuration is refitted on all training data, evaluated with the SAME 5-fold CV and the student-testing
  trips as the untuned model, and saved as artifacts/models/XGBoost_tuned.joblib
"""
import json, time
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from preprocessing.pipeline import FEATURES
from training.train_models import evaluate, SEED

N_TRIALS = 12
SAMPLE_FRAC = 0.25


def make_model(params):
    return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss", **params)


def cv_scores(params, df, pts, n_splits):
    X, y = df[FEATURES].values, df.y.values
    top1, f1, ll, oof = [], [], [], np.zeros(len(df))
    for a, b in GroupKFold(n_splits=n_splits).split(X, y, df.trip.values):
        m = make_model(params).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]; oof[b] = p
        val = df.iloc[b]
        met, _, _ = evaluate(val, pts[pts.trip.isin(val.trip.unique())], p)
        top1.append(met["top1_accuracy"]); f1.append(met["macro_f1"]); ll.append(met["log_loss"])
    return np.mean(top1), np.mean(f1), np.std(f1), np.mean(ll), oof


def main():
    tr = pd.read_parquet("data/processed/train.parquet")
    tr_pts = pd.read_parquet("data/processed/train_points.parquet")
    te = pd.read_parquet("data/processed/student_test.parquet")
    te_pts = pd.read_parquet("data/processed/student_test_points.parquet")

    # fixed point sample for the trials (all candidates of a sampled point are kept)
    rng = np.random.default_rng(SEED)
    keep_pts = tr_pts.sample(frac=SAMPLE_FRAC, random_state=SEED)
    keys = set(zip(keep_pts.trip, keep_pts.point))
    sub = tr[[k in keys for k in zip(tr.trip, tr.point)]].reset_index(drop=True)
    print(f"trial data: {len(sub)} rows, {len(keep_pts)} points", flush=True)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }
        t0 = time.time()
        top1, f1, _, ll, _ = cv_scores(params, sub, keep_pts, n_splits=3)
        trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:2d}: top1={top1:.4f} f1={f1:.4f} ({time.time() - t0:.0f}s) {params}", flush=True)
        return top1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    untuned = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
    study.enqueue_trial({**untuned, "min_child_weight": 1.0, "reg_lambda": 1.0})  # trial 0 = untuned baseline
    t_start = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    search_s = time.time() - t_start
    trials = study.trials_dataframe()
    trials.to_csv("results/metrics/bayes_trials.csv", index=False)
    best = study.best_params
    print(f"best trial {study.best_trial.number}: top1={study.best_value:.4f} {best}  (search {search_s:.0f}s)")

    # refit tuned + untuned on all training data with the same 5-fold CV and the student-testing trips
    rows = []
    for name, params in [("XGBoost (untuned)", untuned), ("XGBoost (tuned)", best)]:
        if name == "XGBoost (untuned)":
            # identical config, data, folds and seed as in training.train_models -> reuse those results
            while True:
                mc = pd.read_csv("results/metrics/model_comparison.csv").set_index("model")
                if "XGBoost" in mc.index:
                    break
                time.sleep(10)  # train_models may still be running in parallel
            r = mc.loc["XGBoost"]
            rows.append({"model": name, "params": json.dumps(params), "cv_top1": r.cv_top1_mean,
                         "cv_macro_f1": r.cv_macro_f1_mean, "cv_macro_f1_std": r.cv_macro_f1_std,
                         "cv_log_loss": r.cv_log_loss_mean, "test_top1": r.test_top1, "test_macro_f1": r.test_macro_f1,
                         "test_log_loss": r.test_log_loss, "test_route_continuity": r.test_route_continuity,
                         "train_time_s": r.train_time_s, "inference_ms_per_point": r.inference_ms_per_point})
            continue
        top1, f1, f1_std, ll, oof = cv_scores(params, tr, tr_pts, n_splits=5)
        m = make_model(params)
        t0 = time.time(); m.fit(tr[FEATURES].values, tr.y.values); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(te[FEATURES].values)[:, 1]; inf_s = time.time() - t0
        met, _, _ = evaluate(te, te_pts, p)
        rows.append({"model": name, "params": json.dumps(params), "cv_top1": top1, "cv_macro_f1": f1,
                     "cv_macro_f1_std": f1_std, "cv_log_loss": ll, "test_top1": met["top1_accuracy"],
                     "test_macro_f1": met["macro_f1"], "test_log_loss": met["log_loss"],
                     "test_route_continuity": met["route_continuity"], "train_time_s": fit_s,
                     "inference_ms_per_point": 1000 * inf_s / len(te_pts)})
        print(rows[-1], flush=True)
        if name == "XGBoost (tuned)":
            joblib.dump(m, "artifacts/models/XGBoost_tuned.joblib")
            np.save("data/processed/oof_XGBoost_tuned.npy", oof)
    pd.DataFrame(rows).to_csv("results/metrics/tuned_vs_untuned.csv", index=False)
    json.dump({"best_params": best, "best_cv_top1_on_sample": study.best_value, "n_trials": N_TRIALS,
               "search_seconds": search_s, "sample_frac": SAMPLE_FRAC, "cv_folds_in_search": 3},
              open("results/metrics/bayes_best.json", "w"), indent=2)


if __name__ == "__main__":
    main()
