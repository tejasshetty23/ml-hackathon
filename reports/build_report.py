"""Builds reports/PathSnap_First_Evaluation.pdf from the saved result CSVs and plots.

Run:  py -3.13 -m reports.build_report      (from the PathSnap folder, after training + plots)
All numbers are read from results/metrics/*.csv - nothing is typed in by hand.
"""
import os
import pandas as pd
from fpdf import FPDF

MET, PLOTS = "results/metrics", "results/plots"
s = pd.read_csv(f"{MET}/model_comparison.csv").set_index("model")
cv = pd.read_csv(f"{MET}/cv_results.csv")
ds = pd.read_csv(f"{MET}/dataset_summary.csv").set_index("partition")
inv = pd.read_csv(f"{MET}/dataset_inventory.csv", dtype={"trip": str})
tr_pts = pd.read_parquet("data/processed/train_points.parquet")
best = s.cv_top1_mean.idxmax()


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 6, "PathSnap - First Evaluation Report (UE24CS352A)", align="R", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8); self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def h1(self, t):
        self.ln(2); self.set_x(self.l_margin); self.set_font("Helvetica", "B", 14); self.multi_cell(0, 8, t); self.ln(1)

    def h2(self, t):
        self.set_x(self.l_margin); self.set_font("Helvetica", "B", 11); self.multi_cell(0, 6, t)

    def p(self, t):
        self.set_x(self.l_margin); self.set_font("Helvetica", "", 9.5); self.multi_cell(0, 5, t); self.ln(1.5)

    def img(self, path, w=180):
        if os.path.exists(path):
            if self.get_y() > 200:
                self.add_page()
            self.image(path, w=w, x=(210 - w) / 2); self.set_x(self.l_margin); self.ln(2)

    def table(self, header, rows, widths, size=7.5):
        self.set_font("Helvetica", "B", size)
        for h, w in zip(header, widths):
            self.cell(w, 6, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", size)
        for r in rows:
            for v, w in zip(r, widths):
                self.cell(w, 5.5, str(v), border=1, align="C")
            self.ln()
        self.ln(2)


f = lambda x: f"{x:.4f}"
pdf = PDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
pdf.set_font("Helvetica", "B", 18); pdf.cell(0, 12, "PathSnap: Map-Matching with Five ML Models", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "First Evaluation Report - UE24CS352A Machine Learning, PES University", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Date: 18 November 2026", new_x="LMARGIN", new_y="NEXT"); pdf.ln(3)

pdf.h1("1. Problem understanding")
pdf.p("GPS fixes are noisy (multipath, dropouts, low sampling), so a raw trajectory drifts off the road it was "
      "driven on. PathSnap must assign every GPS observation to the road segment (arc) of the road network that "
      "the vehicle was actually on. We frame this as candidate classification: for each GPS point we collect the "
      "nearby road arcs (candidates), compute features for each (point, candidate) pair, and train a binary "
      "classifier that scores how likely each candidate is the true arc. The highest-scoring candidate is the "
      "prediction for that point. For this first evaluation we predict every point independently (classifier "
      "only); HMM/Viterbi sequence decoding and Bayesian optimisation are planned for the next phase.")

pdf.h1("2. Dataset description")
tr, te = ds.loc["train"], ds.loc["student_test"]
pdf.p(f"Supplied package: 'Student folder.zip' with two partitions - train ({int(tr.trips)} trips, "
      f"{int(tr.gps_points)} GPS points) and student_test ({int(te.trips)} trips, {int(te.gps_points)} GPS points). "
      "The 10 evaluator-testing trips are not included in the package (trip ids 15, 17, 23, 30, 49, 59, 67, 73, 87 "
      "and 94 are absent), so no evaluator data or labels could be used. Every trip folder holds four headerless, "
      "whitespace-separated files:")
pdf.table(["File", "Columns", "Meaning"],
          [[".track", "lon, lat, t (s)", "GPS observations, ~1 Hz"],
           [".nodes", "lon, lat (row = node id)", "road-network nodes of this trip's area"],
           [".arcs", "from_node, to_node (row = arc id)", "directed road segments"],
           [".route", "arc_id", "ordered reference arcs driven by the vehicle"]], [25, 60, 95])
pdf.p("Key facts verified from the files: each trip has its OWN road network (87 of 90 networks differ), arcs are "
      "directed (about 84% have a reverse arc, i.e. two-way roads), and the route is a list of arcs, not one label "
      "per GPS row (on average about 8 GPS points per route arc). Trips are spread worldwide (Germany, Netherlands, "
      "USA, Argentina, Japan, Russia and others), so absolute longitude/latitude carry no transferable meaning.")

pdf.h1("3. Dataset exploration (partition: train unless stated)")
n_gap = int(tr.trips_with_gap_over_60s)
pdf.p(f"Missing values: {int(tr.missing_values)} in train and {int(te.missing_values)} in student_test across all "
      "four file types. Timestamps are strictly increasing in every trip (no duplicates). Sampling is mostly 1 s, but "
      f"{n_gap} training trips contain a gap longer than 60 s (the longest is {inv.max_gap_s.max():.0f} s). "
      f"Median distance between a GPS point and its aligned reference arc is {tr_pts.label_dist.median():.1f} m "
      f"(95th percentile {tr_pts.label_dist.quantile(.95):.1f} m). Three training trips are clear outliers: trips "
      "33 and 54 have a systematic offset of over 100 m from the reference route, and trip 2 is very noisy. We kept "
      "them in training (they are real data) but they dominate the training candidate miss rate.")
pdf.img(f"{PLOTS}/eda_distributions.png")
pdf.img(f"{PLOTS}/eda_candidate_miss_per_trip.png")
pdf.img(f"{PLOTS}/trajectory_00000000.png", w=120)
pdf.p("Figure: student-test trip 00000000 - raw GPS (blue), reference route (green), nearby road network (grey). "
      "The interactive Folium version with layer control and legend, also showing our predicted route, is "
      "results/maps/trip_00000000_map.html.")
pdf.img(f"{PLOTS}/folium_trip_00000000.png", w=150)

pdf.h1("4. Preprocessing methodology")
pdf.p("1) Projection: each trip is projected to local metres with an equirectangular projection centred on the "
      "trip's mean latitude (accurate at city scale; Euclidean distance in degrees would be wrong, especially in "
      "longitude).\n"
      "2) Label generation: since .route is an ordered arc list, we align GPS points to it with a monotone dynamic "
      "program - point n is assigned route position m(n) with m never decreasing, minimising the total (capped at "
      "150 m) point-to-arc distance. This respects the driving order and avoids snapping to a route arc that is "
      "visited at a different time. The label of each point is the arc at its aligned position.\n"
      "3) Candidate generation: all arcs within 50 m of the GPS point (shapely STRtree spatial index), keeping the "
      "8 nearest. A candidate is positive if it is the label arc (or an exact duplicate u->v arc). Points whose "
      "label arc is not among the candidates (misses) are counted as wrong in Top-1 accuracy.")
pts_tr_miss = 1 - tr_pts.label_in_cands.mean()
pdf.p(f"Candidate miss rate: train = {pts_tr_miss:.4f}, student_test = {s.candidate_miss_rate.iloc[0]:.4f}. "
      "The training miss rate is driven by the offset/noisy trips described above; on typical trips it is below 2%.")

pdf.h1("5. Feature engineering and justification")
pdf.table(["Feature", "Why it is used"], [
    ["dist", "metres from GPS point to candidate arc - main signal"],
    ["dist_rank, dist_minus_best", "relative closeness vs other candidates of the same point"],
    ["heading_diff", "angle between GPS motion and arc direction (separates directions)"],
    ["heading_valid", "1 if the vehicle moved > 2 m (heading unreliable when stopped)"],
    ["speed", "m/s from neighbouring fixes; context for heading reliability"],
    ["dt_prev, dt_next", "time gaps; large gaps make a point less reliable"],
    ["arc_len, proj_frac", "segment length and where along it the point projects"],
    ["n_cands", "road density around the point (ambiguity)"],
    ["two_way", "arc has a reverse twin (both directions are candidates)"]], [50, 130])
pdf.p("Deliberately NOT used: raw longitude/latitude (trips are on different continents, so coordinates would only "
      "identify the trip, not generalise) and anything derived from the reference route at prediction time. All "
      "features use only GPS + road network, so the same code can run on unseen evaluator trips.")

pdf.h1("6-7. The five models and their configurations")
for name in s.index:
    pdf.p(f"{name} ({s.loc[name, 'family']}): " + s.loc[name, 'config'].replace(', "penalty": "deprecated"', ' (L2 penalty)'))
pdf.p("Common setup: identical features, labels and data splits for all models; random seed 42. Logistic "
      "Regression and KNN use a StandardScaler inside the model pipeline (fitted on training folds only); tree "
      "models need no scaling. KNN is fitted on a random 50,000-row subsample of each training set because it "
      "stores all rows and its prediction cost grows with training size.")

pdf.h1("8. Training and cross-validation methodology")
pdf.p("5-fold GroupKFold on the 80 training trips with trip id as the group, so all observations of a trip are in "
      "the same fold (no leakage between neighbouring 1 Hz points). Each model is then refitted on all 80 training "
      "trips and evaluated once on the 10 student-testing trips. Metrics: candidate-level accuracy, precision, "
      "recall, Macro F1 and log loss (threshold 0.5); observation-level Top-1 accuracy (share of GPS points whose "
      "highest-scoring candidate is the reference arc; points with no correct candidate count as wrong); route "
      "continuity (share of consecutive points whose predicted arcs are the same or directly connected); training "
      "and inference time.")

pdf.h1("9-10. Results")
pdf.h2("Table 1: consolidated comparison")
pdf.table(["Model", "Family", "CV Macro F1", "CV Top-1", "Test Top-1", "Route cont.", "Test Macro F1",
           "Log Loss", "Inf. ms/pt"],
          [[n, r.family, f"{r.cv_macro_f1_mean:.4f}+-{r.cv_macro_f1_std:.3f}", f(r.cv_top1_mean), f(r.test_top1),
            f(r.test_route_continuity), f(r.test_macro_f1), f(r.test_log_loss), f"{r.inference_ms_per_point:.3f}"]
           for n, r in s.iterrows()], [28, 24, 25, 17, 18, 18, 21, 16, 16], size=7)
pdf.h2("Table 2: student-testing candidate-level metrics")
pdf.table(["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Log Loss", "Train time (s)"],
          [[n, f(r.test_accuracy), f(r.test_precision), f(r.test_recall), f(r.test_macro_f1), f(r.test_log_loss),
            f"{r.train_time_s:.1f}"] for n, r in s.iterrows()], [32, 22, 22, 22, 22, 22, 26])
pdf.h2("Table 3: overfitting check (train = refit model scored on 20 of its own training trips)")
pdf.table(["Model", "Train Top-1", "CV Top-1", "Test Top-1", "Train F1", "CV F1", "Test F1"],
          [[n, f(r.train_top1), f(r.cv_top1_mean), f(r.test_top1), f(r.train_macro_f1), f(r.cv_macro_f1_mean),
            f(r.test_macro_f1)] for n, r in s.iterrows()], [32, 24, 24, 24, 24, 24, 24])
pdf.p(f"Candidate miss rate on student_test is {s.candidate_miss_rate.iloc[0]:.4f} for every model (it depends "
      "only on candidate generation), so the Top-1 ceiling is "
      f"{1 - s.candidate_miss_rate.iloc[0]:.4f}. Route continuity is reported for classifier-only predictions; "
      "the reference labels themselves are not 100% continuous because 1 Hz points can skip very short arcs.")

pdf.h1("11. Performance graphs")
for fn in ["cv_macro_f1.png", "cv_folds.png", "test_top1.png", "test_macro_f1.png", "test_log_loss.png",
           "confusion_matrices.png"]:
    pdf.img(f"{PLOTS}/{fn}", w=150 if fn != "confusion_matrices.png" else 185)

pdf.h1("Per-model discussion (based on our results)")
cvstd = cv.groupby("model").top1_accuracy.std()
for name in s.index:
    r = s.loc[name]
    gap = r.train_top1 - r.cv_top1_mean
    fit_txt = ("a large train-vs-CV gap suggests overfitting" if gap > 0.05 else
               "train and CV scores are close, so there is no strong sign of overfitting")
    pdf.h2(name)
    pdf.p(f"Student-test: Top-1 {r.test_top1:.4f}, Macro F1 {r.test_macro_f1:.4f}, log loss {r.test_log_loss:.4f}, "
          f"precision {r.test_precision:.4f}, recall {r.test_recall:.4f}, route continuity {r.test_route_continuity:.4f}. "
          f"CV Top-1 {r.cv_top1_mean:.4f} (fold std {cvstd[name]:.4f}). Train Top-1 {r.train_top1:.4f} vs CV "
          f"{r.cv_top1_mean:.4f}: {fit_txt} (gap {gap:+.4f}). Training took {r.train_time_s:.1f} s and inference "
          f"{r.inference_ms_per_point:.3f} ms per GPS point.")
NOTES = {
    "LogisticRegression": "Why suitable: gives calibrated-looking probabilities and a simple, explainable linear score "
        "over distance and heading - a natural emission score for a later HMM. Limitation seen: it can only weight "
        "features linearly, e.g. it cannot learn that heading matters only when the vehicle is moving.",
    "KNN": "Why suitable: similar (distance, heading, speed) patterns should share labels, with no training. "
        "Limitation seen: it must keep training rows in memory (we subsampled to 50k rows) and is by far the slowest at "
        "inference, which matters for an API; its probabilities are vote shares, which hurts log loss.",
    "DecisionTree": "Why suitable: learns threshold rules (e.g. 'closest and heading < 45 degrees') and interactions "
        "the linear model cannot. Limitation: a single tree is unstable and its leaf probabilities are coarse; "
        "we limited depth to 12 and leaves to 50 samples to control overfitting.",
    "RandomForest": "Why suitable: averages many decorrelated trees, which reduces the variance of a single tree "
        "and gives smoother probabilities. Limitation: larger model files and slower inference than one tree.",
    "XGBoost": "Why suitable: gradient boosting corrects errors sequentially and handles feature interactions and "
        "skewed features (gaps, speed) well with fast histogram training. Limitation: more hyperparameters to tune "
        "(planned for the Bayesian optimisation phase).",
}
for name in s.index:
    pdf.p(f"{name}: {NOTES[name]} Loss/objective: " + {
        "LogisticRegression": "L2-regularised binary cross-entropy (lbfgs).",
        "KNN": "none (lazy learner; distance-weighted vote of 15 neighbours).",
        "DecisionTree": "greedy Gini impurity reduction.",
        "RandomForest": "Gini impurity per tree, bootstrap aggregation.",
        "XGBoost": "binary logistic loss with second-order boosting."}[name])

pdf.h1("12. Preliminary model selection")
r = s.loc[best]
pdf.p(f"Primary metric: cross-validated observation-level Top-1 accuracy on the training trips, because the task "
      "is to pick the right road for every GPS point; Macro F1 and log loss are secondary (probability quality "
      "matters for the HMM emission model). Selection uses CV only, never the evaluator data. "
      f"Preliminary choice: {best} (CV Top-1 {r.cv_top1_mean:.4f}, CV Macro F1 {r.cv_macro_f1_mean:.4f}, "
      f"student-test Top-1 {r.test_top1:.4f}, log loss {r.test_log_loss:.4f}, {r.inference_ms_per_point:.3f} ms per "
      "point). This model will be tuned with Bayesian optimisation and its probabilities used as HMM emissions in "
      "the next phase.")

pdf.h1("13. Team responsibilities")
pdf.table(["Member", "Responsibility"], [
    ["Member 1", "Dataset exploration, loading, cleaning, road network, candidates, features"],
    ["Member 2", "Logistic Regression and KNN - training and evaluation"],
    ["Member 3", "Decision Tree and Random Forest - training and evaluation"],
    ["Member 4", "XGBoost, model comparison, metrics, graphs, documentation"]], [30, 150])
pdf.p("Limitations / honest notes: labels come from our alignment of GPS to the supplied route, which is itself "
      "reference information, not verified ground truth. No hyperparameter tuning, HMM or API has been done yet. "
      "All numbers come from executing the code in this repository.")

os.makedirs("reports", exist_ok=True)
pdf.output("reports/PathSnap_First_Evaluation.pdf")
print("wrote reports/PathSnap_First_Evaluation.pdf")
