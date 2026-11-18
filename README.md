# PathSnap — Map-matching and route reconstruction

Noisy GPS points are matched to the road segments (arcs) the vehicle actually drove on.
For every GPS point we collect nearby road arcs (candidates), compute features, and a classifier scores each
candidate. An HMM with Viterbi decoding then picks the most likely *connected* sequence of arcs. A FastAPI backend
(Laptop 1) runs the whole pipeline; a browser UI (Laptop 2, plain HTML + JavaScript) lets the evaluator pick a trip and shows the result.

## 1. Installation
```
py -3.13 -m pip install -r requirements.txt
```
Unzip `Student folder.zip` so the data is at `data/raw/train/<trip>/` and `data/raw/student_test/<trip>/`.

## 2. Reproduce all experiments (run from this folder, in order)
| Step | Command | Output |
|---|---|---|
| 1. Cleaning, labels, candidates, features | `py -3.13 -m preprocessing.pipeline` | `data/processed/`, `results/metrics/cleaning_report.csv` |
| 2. Train + compare 5 models | `py -3.13 -m training.train_models` | `artifacts/models/*.joblib`, `results/metrics/model_comparison.csv` |
| 3. Bayesian optimisation (XGBoost) | `py -3.13 -m training.tune_bayes` | `XGBoost_tuned.joblib`, `bayes_trials.csv`, `tuned_vs_untuned.csv` |
| 4. HMM decoder tuning + comparison | `py -3.13 -m evaluation.hmm_eval` | `artifacts/final_config.json`, `hmm_comparison.csv`, `hmm_per_trip.csv` |
| 5. XGBoost loss curve | `py -3.13 -m training.xgb_learning_curve` | `results/plots/xgb_learning_curve.png` |
| 6. Plots + Folium maps | `py -3.13 -m visualisation.make_plots` | `results/plots/`, `results/maps/` |
| 7. Tests | `py -3.13 -m tests.test_models_load` | saved models load, pipeline deterministic |
| 8. PDF report | `py -3.13 -m reports.build_report` | `reports/PathSnap_Technical_Report.pdf` |

Data partitions: 80 training trips (fitting, cross-validation, tuning), 10 student-testing trips (model comparison).
Evaluator-testing trips are never used for training, tuning or model selection.

## 3. Deployment (two laptops)
**Laptop 1 — API** (needs this repo, the trained `artifacts/`, and the evaluation trips):
```
# put the evaluator trips at data/evaluator_test/<trip>/<trip>.track|.nodes|.arcs|.route
set PATHSNAP_DATA_DIR=data/evaluator_test
py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Without `PATHSNAP_DATA_DIR` the API serves the 10 student-testing trips. Every request is logged in this window
(show these logs to the evaluator). Find the laptop's IP with `ipconfig` and allow port 8000 in the Windows firewall.

**Laptop 2 — UI** (needs only the file `ui/index.html` and a browser; Python is optional):
```
py -3.13 -m http.server 8501 --directory ui
```
Open `http://localhost:8501/?api=http://<laptop-1-ip>:8000` (or double-click `ui/index.html` and type the API
address in the box). The page lists the trips from the API, sends the selected trip id to `POST /predict/{trip_id}`
and draws the predicted route (Leaflet map) with the accuracy computed by the API. Ground truth never leaves Laptop 1.
On a single laptop just open `http://localhost:8501` (default API address `http://localhost:8000`).

API endpoints: `GET /health`, `GET /trips`, `POST /predict/{trip_id}`.

## 4. Folders
| Folder | Content |
|---|---|
| `preprocessing/pipeline.py` | loading, cleaning, projection, candidates, features, route alignment (labels) |
| `training/` | `train_models.py` (5 models), `tune_bayes.py` (Optuna), `xgb_learning_curve.py` |
| `models/hmm.py` | HMM + Viterbi decoder |
| `evaluation/hmm_eval.py` | decoder tuning, classifier-only vs HMM comparison |
| `api/main.py`, `ui/index.html` | FastAPI backend, browser UI (HTML + JavaScript + Leaflet) |
| `visualisation/make_plots.py` | EDA plots, comparison graphs, Folium maps |
| `reports/build_report.py` | builds the PDF from the result files |
| `artifacts/` | saved models + `final_config.json` (model file + HMM parameters used by the API) |
| `results/` | metrics CSVs, plots, maps, logs |
