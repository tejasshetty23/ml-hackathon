# Team Contributions — PathSnap (First Evaluation)

All four members must be able to explain the full pipeline:
raw files → monotone route alignment (labels) → 50 m / top-8 candidates → 12 features → 5 classifiers → Top-1 per GPS point.

## Member 1 — Data and preprocessing
- Dataset exploration and inventory (`visualisation/make_plots.py: dataset_summary`, `results/metrics/dataset_*.csv`)
- Data loading (`preprocessing/pipeline.py: load_trip`)
- Cleaning / checks: missing values, timestamp order, time gaps, outlier trips (2, 33, 54)
- Road network: nodes + directed arcs per trip, two-way arcs
- Candidate generation (shapely STRtree, 50 m radius, 8 nearest) and candidate miss rate
- Label generation: monotone alignment of GPS points to the `.route` arc sequence (`align_to_route`)
- Feature engineering (`process_trip`)

## Member 2 — Logistic Regression and KNN
- `MODELS["LogisticRegression"]`, `MODELS["KNN"]` in `training/train_models.py`
- Scaling (StandardScaler inside the pipeline), KNN subsampling (50k rows) and why
- Explain CV results, student-test results, log loss and inference time for both

## Member 3 — Decision Tree and Random Forest
- `MODELS["DecisionTree"]`, `MODELS["RandomForest"]`
- Depth / leaf-size limits and overfitting check (train vs CV vs test table in the report)
- Explain confusion matrices for both models

## Member 4 — XGBoost and comparison
- `MODELS["XGBoost"]`
- Metric code (`evaluate`, `route_continuity`), comparison CSVs in `results/metrics/`
- Graphs and Folium map (`visualisation/make_plots.py`), PDF report (`reports/build_report.py`)
- Preliminary model selection (primary metric: CV observation-level Top-1 accuracy)
