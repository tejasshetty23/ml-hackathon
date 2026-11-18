# PathSnap — Map-matching with ML (First Evaluation)

Each GPS point is matched to the road segment (arc) the vehicle was on. For every GPS point we collect the
nearby road arcs (candidates), compute features, and five classifiers score each candidate. The highest score wins.

## Setup
```
py -3.13 -m pip install -r requirements.txt
```
Unzip `Student folder.zip` so that the data sits at `data/raw/train/<trip>/` and `data/raw/student_test/<trip>/`.

## Run (from this folder, in order)
```
py -3.13 -m preprocessing.pipeline        # labels + candidates + features -> data/processed/
py -3.13 -m training.train_models         # 5 models: GroupKFold CV + student-test eval -> artifacts/, results/metrics/
py -3.13 -m visualisation.make_plots      # EDA + comparison graphs + Folium map -> results/plots, results/maps
py -3.13 -m tests.test_models_load        # saved models load and work
py -3.13 -m reports.build_report          # reports/PathSnap_First_Evaluation.pdf
```

## Folders
| Folder | Content |
|---|---|
| `preprocessing/pipeline.py` | loading, projection to metres, route alignment (labels), candidates, features |
| `training/train_models.py` | the 5 models, cross-validation, student-test evaluation, metrics |
| `visualisation/make_plots.py` | EDA plots, comparison graphs, confusion matrices, Folium map |
| `reports/build_report.py` | builds the PDF from the result CSVs |
| `artifacts/models/` | saved models (`*.joblib`) |
| `results/metrics/` | `model_comparison.csv`, `cv_results.csv`, `student_test_results.csv`, dataset tables |
| `results/plots/`, `results/maps/` | graphs and the interactive Folium map |
| `results/logs/` | console logs of the runs |

## Models
Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost (seed 42). Not done yet (next phase):
Bayesian optimisation, HMM/Viterbi, FastAPI, UI.
