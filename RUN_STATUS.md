# RUN_STATUS — PathSnap (18 Nov 2026)

## Completed (all executed, all numbers from `results/metrics/`)
| Component | Status | Evidence |
|---|---|---|
| Dataset exploration | done | `dataset_inventory.csv`, `dataset_summary.csv`, `results/plots/eda_*.png` |
| Cleaning | done | `cleaning_report.csv`: 1,397 self-loop/zero-length arcs excluded; 20,550 training points (9.9%) with label > 100 m removed (train only); 0 invalid GPS rows |
| Labels, candidates, features | done | train candidate miss rate 0.119 → 0.023 after cleaning; student_test 0.004 |
| 5 models, 5-fold GroupKFold + student-test | done | `model_comparison.csv`, `cv_results.csv`, `student_test_results.csv` |
| Bayesian optimisation (Optuna TPE, 12 trials) | done | `bayes_trials.csv`, `bayes_best.json`, `tuned_vs_untuned.csv`, `plots/bayes_history.png` |
| HMM + Viterbi | done | `hmm_comparison.csv`, `hmm_per_trip.csv`, `hmm_grid.csv` |
| XGBoost train/val loss curve | done | `xgb_learning_curve.csv`, `plots/xgb_learning_curve.png` |
| Folium maps (all 10 student-test trips) | done | `results/maps/trip_<id>_map.html` |
| FastAPI backend | done, tested | `api/main.py`, log `results/logs/api_server.log` |
| Streamlit UI | done, tested | `ui/app.py`, `plots/ui_screenshot.png` |
| Tests | pass | `results/logs/tests.log` |
| Technical report | done | `reports/PathSnap_Technical_Report.pdf` (6 pages) |

## Results — classifiers (student-test = 10 trips; CV = 5-fold GroupKFold on 80 cleaned training trips)
| Model | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Log loss | Continuity | Infer ms/pt |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.879 | 0.818 | 0.839 | 0.888 | 0.175 | 0.912 | 0.001 |
| KNN | 0.892 | 0.817 | 0.843 | 0.902 | 0.213 | 0.909 | 1.721 |
| Decision Tree | 0.900 | 0.828 | 0.851 | 0.908 | 0.130 | 0.921 | 0.002 |
| Random Forest | 0.903 | 0.830 | 0.854 | 0.910 | 0.122 | 0.920 | 0.083 |
| **XGBoost** | **0.904** | **0.832** | **0.858** | **0.912** | **0.119** | 0.923 | 0.050 |

Bayesian tuning: CV Top-1 0.8321 (untuned) vs 0.8315 (tuned) → no accuracy gain; tuned model is ~6x faster.
Final classifier = untuned XGBoost (selected on CV Top-1).

## Results — route reconstruction (student-test)
| Method | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| XGBoost only | 0.858 | 0.923 | 0.847 |
| **XGBoost + HMM** (beta 5, w_e 0.5) | **0.947** | **0.982** | **0.928** |

HMM improved all 10 student-test trips. Evaluator trips were not used anywhere.

## Artifacts
- `artifacts/models/{LogisticRegression,KNN,DecisionTree,RandomForest,XGBoost,XGBoost_tuned}.joblib`
- `artifacts/final_config.json` — model file + HMM parameters loaded by the API

## Reproduce / deploy
See README.md (sections 2 and 3). Full order: `preprocessing.pipeline` → `training.train_models` →
`training.tune_bayes` → `evaluation.hmm_eval` → `training.xgb_learning_curve` → `visualisation.make_plots` →
`tests.test_models_load` → `reports.build_report`.

## Known issues / honest notes
- Bayesian tuning ran in parallel with training of Random Forest and XGBoost to save time, so the training times of
  those two models (112 s, 81 s) and the first tuning trials are inflated by CPU sharing. LR, KNN and Decision Tree
  timings, and all accuracy/F1/log-loss values, are unaffected (fixed seeds).
- Tuning trials use a 25% sample of GPS points and 3-fold CV (time budget); the tuned-vs-untuned comparison uses the
  full data and the same 5-fold CV.
- KNN is fitted on a 50,000-row sample.
- Labels come from our alignment of GPS to the supplied route (reference information, not verified ground truth).
- Demo day: copy the evaluator trips to `data/evaluator_test/` on Laptop 1 and set `PATHSNAP_DATA_DIR` (README §3).
  Take the required screenshots of the API console during the demo.
