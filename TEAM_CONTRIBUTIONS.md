# Team Contributions — PathSnap

All four members must be able to explain the full pipeline:
raw files → cleaning → monotone route alignment (labels) → 50 m / top-8 candidates → 12 features → classifier
→ HMM/Viterbi → FastAPI → Streamlit UI.

## Member 1 — Data and preprocessing
- Dataset exploration and inventory (`visualisation/make_plots.py: dataset_summary`, `results/metrics/dataset_*.csv`)
- Loading and cleaning (`preprocessing/pipeline.py: read_trip_files, clean_track`; road-network cleaning of
  self-loops/zero-length arcs; training-only removal of points with label distance > 100 m → `cleaning_report.csv`)
- Candidate generation (STRtree, 50 m, 8 nearest) and candidate miss rate
- Label generation: monotone alignment of GPS points to the `.route` arc sequence (`align_to_route`)
- Feature engineering (`candidates_and_features`)

## Member 2 — Logistic Regression, KNN and the API
- `MODELS["LogisticRegression"]`, `MODELS["KNN"]` in `training/train_models.py` (scaling, KNN 50k-row sample)
- FastAPI backend `api/main.py`: endpoints, model loading, server-side evaluation, request logging

## Member 3 — Decision Tree, Random Forest and the HMM
- `MODELS["DecisionTree"]`, `MODELS["RandomForest"]`; overfitting check (seen vs unseen trips)
- HMM/Viterbi decoder `models/hmm.py` and its evaluation `evaluation/hmm_eval.py`
  (states, emissions, transitions, gaps, disconnected roads, decoder-parameter tuning)

## Member 4 — XGBoost, optimisation, comparison and UI
- `MODELS["XGBoost"]`, Bayesian optimisation `training/tune_bayes.py`, learning curve `training/xgb_learning_curve.py`
- Metrics, comparison CSVs, graphs, Folium maps (`visualisation/make_plots.py`)
- Streamlit UI `ui/app.py`, PDF report `reports/build_report.py`
