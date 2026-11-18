# RUN_STATUS — PathSnap first evaluation (18 Nov 2026)

## Completed
- Dataset inspected: train = 80 trips / 208,417 GPS points; student_test = 10 trips / 19,160 points.
  Evaluator-testing trips are **not** in the supplied zip (ids 15, 17, 23, 30, 49, 59, 67, 73, 87, 94 missing) — nothing of it was used.
- Preprocessing: per-trip local metric projection, monotone alignment of GPS points to `.route` (labels),
  candidates = arcs within 50 m, 8 nearest (shapely STRtree), 12 features. 1,085,227 train rows / 113,694 test rows.
- Candidate miss rate: **train 0.1192**, **student_test 0.0041** (train is dominated by trips 2, 33, 54 which are offset/noisy).
- 5 models trained with 5-fold GroupKFold (by trip) on train, refit on all train, evaluated on student_test, saved.
- Saved models reload and predict (`tests/test_models_load.py`, log in `results/logs/tests.log`).
- Plots, confusion matrices, Folium map, PDF report, TEAM_CONTRIBUTIONS.md.

## Results (from `results/metrics/model_comparison.csv`)
| Model | CV Macro F1 (mean ± std) | CV Top-1 | Test Top-1 | Test Macro F1 | Test Precision | Test Recall | Test Log Loss | Test Route continuity | Inference ms/point |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8735 ± 0.033 | 0.7459 | 0.8381 | 0.8871 | 0.8215 | 0.8020 | 0.1754 | 0.9111 | 0.001 |
| KNN | 0.8839 ± 0.032 | 0.7439 | 0.8427 | 0.9026 | 0.8589 | 0.8164 | 0.2231 | 0.9134 | 0.840 |
| Decision Tree | 0.8939 ± 0.032 | 0.7532 | 0.8515 | 0.9086 | 0.8526 | 0.8429 | 0.1292 | 0.9205 | 0.001 |
| Random Forest | 0.8975 ± 0.031 | 0.7566 | 0.8531 | 0.9096 | 0.8799 | 0.8193 | 0.1232 | 0.9197 | 0.022 |
| **XGBoost** | **0.8993 ± 0.030** | **0.7587** | **0.8546** | **0.9111** | 0.8680 | 0.8356 | **0.1198** | **0.9233** | 0.020 |

Training time (clean sequential run, 8-core laptop, nothing else running):

| Model | Mean CV fold fit (s) | Final fit on all 80 trips (s) |
|---|---|---|
| Logistic Regression | 2.9 | 2.8 |
| KNN (50k-row sample, lazy learner) | 0.06 | 0.04 |
| Decision Tree | 9.4 | 11.6 |
| Random Forest | 35.4 | 54.0 |
| XGBoost | 27.1 | 37.8 |

- Reference-route continuity on student_test (ceiling of the continuity metric): 0.9848.
- Top-1 ceiling on student_test = 1 − miss rate = 0.9959.
- Preliminary selection: **XGBoost** (best CV Top-1 = primary metric, best CV Macro F1, lowest log loss).
- CV Top-1 is lower than test Top-1 because CV folds 3 and 4 contain the offset/noisy training trips (Top-1 ≈ 0.58–0.67 in those folds).

## Artifacts
- Models: `artifacts/models/{LogisticRegression,KNN,DecisionTree,RandomForest,XGBoost}.joblib`
- Metrics: `results/metrics/model_comparison.csv`, `cv_results.csv`, `student_test_results.csv`,
  `dataset_inventory.csv`, `dataset_summary.csv`, `confusion_*.npy`, `student_test_predictions_*.csv`
- Plots: `results/plots/` · Map: `results/maps/trip_00000000_map.html` · Report: `reports/PathSnap_First_Evaluation.pdf`
- Logs: `results/logs/preprocessing.log`, `training.log` (all 5 models, sequential run), `plots.log`, `tests.log`

## Reproduce (from `PathSnap/`)
```
py -3.13 -m pip install -r requirements.txt
py -3.13 -m preprocessing.pipeline
py -3.13 -m training.train_models            # or a subset, e.g.: py -3.13 -m training.train_models KNN
py -3.13 -m visualisation.make_plots
py -3.13 -m tests.test_models_load
py -3.13 -m reports.build_report
```

## Known issues / honest notes
- Timings come from a clean sequential re-run (all 5 models one after another). An earlier run had KNN running in
  parallel and inflated times; the re-run reproduced every accuracy/F1/log-loss value exactly (fixed seeds).
- KNN is fitted on a 50,000-row random subsample (brute-force search) for speed.
- The "train" overfitting check scores each refit model on 20 random training trips, not all 80.
- Labels come from our alignment to the supplied route (reference information, not verified ground truth).
- Duplicate u→v arcs mean a few GPS points have 2 positive candidates.
- Not done yet (next phase, by design): Bayesian hyperparameter optimisation, HMM/Viterbi, FastAPI, UI, GitHub repo.
