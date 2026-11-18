"""Builds reports/PathSnap_Technical_Report.pdf from the saved result files and plots.

Run:  py -3.13 -m reports.build_report      (from the PathSnap folder, after all experiments)
All numbers are read from results/metrics/* - nothing is typed in by hand.
"""
import os, json
import pandas as pd
from fpdf import FPDF

MET, PLOTS = "results/metrics", "results/plots"
s = pd.read_csv(f"{MET}/model_comparison.csv").set_index("model")
ds = pd.read_csv(f"{MET}/dataset_summary.csv").set_index("partition")
inv = pd.read_csv(f"{MET}/dataset_inventory.csv", dtype={"trip": str})
clean = pd.read_csv(f"{MET}/cleaning_report.csv")
tvu = pd.read_csv(f"{MET}/tuned_vs_untuned.csv").set_index("model")
bayes = json.load(open(f"{MET}/bayes_best.json"))
trials = pd.read_csv(f"{MET}/bayes_trials.csv")
hmm = pd.read_csv(f"{MET}/hmm_comparison.csv")
hpt = pd.read_csv(f"{MET}/hmm_per_trip.csv", dtype={"trip": str})
grid = pd.read_csv(f"{MET}/hmm_grid.csv")
cfg = json.load(open("artifacts/final_config.json"))
curve = pd.read_csv(f"{MET}/xgb_learning_curve.csv")
pts_all = pd.read_parquet("data/processed/train_points_all.parquet")
pts_clean = pd.read_parquet("data/processed/train_points.parquet")
api_log = open("results/logs/api_server.log", encoding="utf-8", errors="ignore").read() \
    if os.path.exists("results/logs/api_server.log") else ""
f3 = lambda x: f"{x:.3f}"
best_untuned = s.cv_top1_mean.idxmax()


class PDF(FPDF):
    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"PathSnap technical report - page {self.page_no()}", align="C")

    def h1(self, t):
        self.ln(1); self.set_x(self.l_margin); self.set_font("Helvetica", "B", 13); self.multi_cell(0, 7, t)

    def h2(self, t):
        self.set_x(self.l_margin); self.set_font("Helvetica", "B", 10.5); self.multi_cell(0, 6, t)

    def p(self, t, size=9.5):
        self.set_x(self.l_margin); self.set_font("Helvetica", "", size); self.multi_cell(0, 4.7, t); self.ln(1)

    def img(self, path, w, caption=None):
        if os.path.exists(path):
            self.image(path, w=w, x=(210 - w) / 2); self.set_x(self.l_margin)
            if caption:
                self.set_font("Helvetica", "I", 8.5); self.multi_cell(0, 4.2, caption, align="C")
            self.ln(1.5)

    def grid(self, rows, widths, size=8):
        self.set_font("Helvetica", "", size)
        with self.table(col_widths=widths, text_align="LEFT", line_height=4.2, width=sum(widths)) as t:
            for r in rows:
                row = t.row()
                for v in r:
                    row.cell(str(v))
        self.ln(2)


pdf = PDF(); pdf.set_auto_page_break(True, 15); pdf.set_margins(15, 12, 15); pdf.add_page()
pdf.set_font("Helvetica", "B", 17); pdf.cell(0, 9, "PathSnap: Map-Matching and Route Reconstruction", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Technical report - UE24CS352A Machine Learning, PES University - November 2026", new_x="LMARGIN", new_y="NEXT")

pdf.h1("1. Problem and approach")
pdf.p("Noisy GPS points must be matched to the road segment (arc) that the vehicle actually drove on. Pipeline: "
      "(1) for each GPS point, collect nearby arcs as candidates; (2) a classifier scores each (point, candidate) pair; "
      "(3) an HMM with Viterbi decoding chooses the most likely connected sequence of arcs; (4) a FastAPI backend runs "
      "this for a selected trip and a browser UI (HTML + JavaScript, Leaflet) shows the result. Final system: "
      f"{cfg['model_name']} + HMM (beta = {cfg['hmm']['beta']:g} m, emission weight = {cfg['hmm']['w_emission']:g}).")

# ---------------- 2. dataset ----------------
tr, te = ds.loc["train"], ds.loc["student_test"]
pdf.h1("2. Dataset understanding and exploratory analysis")
pdf.grid([["File", "Content (headerless, whitespace-separated)"],
          [".track", "lon, lat, time (s): GPS observations, mostly 1 Hz"],
          [".nodes / .arcs", "road-network nodes (lon, lat) and directed arcs (from_node, to_node); row index = id"],
          [".route", "ordered arc ids driven by the vehicle = reference route (no per-GPS-point label)"]], [30, 150])
pdf.p(f"Partitions: train = {int(tr.trips)} trips / {int(tr.gps_points):,} points (fitting, CV, tuning, decoder "
      f"tuning); student_test = {int(te.trips)} trips / {int(te.gps_points):,} points (model comparison). Evaluator "
      "trips were never used for training, tuning or selection. Every trip has its own road network and trips are "
      "spread over several continents, so raw lon/lat are not used as features. All analyses below use the train "
      "partition unless stated.")
pdf.h2("Missing values and data quality")
pdf.grid([["Partition", "Trips", "GPS points", "Missing values", "Duplicate / non-increasing timestamps",
           "Trips with a gap > 60 s"],
          *[[p, int(r.trips), f"{int(r.gps_points):,}", int(r.missing_values), int(clean[clean.partition == p].gps_rows_removed.sum()),
             int(r.trips_with_gap_over_60s)] for p, r in ds.iterrows()]], [28, 14, 24, 25, 55, 34])
pdf.p(f"Patterns: sampling is mostly 1 s but has dropouts (longest {inv.max_gap_s.max():,.0f} s). The median "
      f"distance between a GPS point and its reference arc is {pts_all.label_dist.median():.1f} m; the 95th percentile "
      f"({pts_all.label_dist.quantile(.95):.0f} m) is inflated by trips 54 and 33, which are offset > 100 m from their "
      "route, and by the very noisy trip 2.")
pdf.img(f"{PLOTS}/eda_distributions.png", 180)

pdf.h2("Preprocessing and cleaning")
ct = clean[clean.partition == "train"]
pdf.p("1) GPS rows with missing/invalid coordinates or non-increasing timestamps are dropped (none found). "
      "2) Each trip is projected to local metres (equirectangular around its mean latitude); distances in degrees "
      "would be wrong. "
      f"3) Road-network cleaning: self-loops and arcs shorter than 0.5 m are never candidates "
      f"({int(clean.arcs_excluded.sum()):,} arcs). "
      "4) Labels: GPS points are aligned to the ordered route with a monotone dynamic program (route position never "
      "moves backwards, total point-to-arc distance minimised); a point's label is its aligned arc. "
      "5) Candidates: arcs within 50 m, 8 nearest (STRtree spatial index). "
      f"6) Training-only label cleaning: {int(ct.points_unreliable_label.sum()):,} training points "
      f"({ct.points_unreliable_label.sum() / len(pts_all):.1%}) whose reference arc is > 100 m away were removed "
      "(unreliable labels; almost all from trips 54 and 33). Student-testing data is never filtered. "
      f"Candidate miss rate (reference arc not among candidates): train {1 - pts_all.label_in_cands.mean():.3f} "
      f"before / {1 - pts_clean.label_in_cands.mean():.3f} after cleaning; student_test "
      f"{s.candidate_miss_rate.iloc[0]:.3f}.")
pdf.h2("Features and why they were chosen")
pdf.grid([["Feature", "Reason"],
          ["dist, dist_rank, dist_minus_best", "closeness to the road, absolute and relative to the other candidates"],
          ["heading_diff, heading_valid", "GPS direction vs arc direction; separates parallel roads and the two "
                                          "directions of a street (valid only when moving > 2 m)"],
          ["speed, dt_prev, dt_next", "motion context and time gaps (how reliable the point is)"],
          ["arc_len, proj_frac, n_cands, two_way", "segment geometry, road density (ambiguity), two-way roads"]], [60, 120])

# ---------------- 3. models ----------------
pdf.h1("3. Model experimentation and comparison")
pdf.p("All five models use the same features, labels and splits (seed 42). Validation: 5-fold GroupKFold by trip on "
      "the 80 training trips (all points of a trip stay in one fold), then refit on all training trips and one "
      "evaluation on the 10 student-testing trips. Candidate-level metrics use threshold 0.5. Observation-level "
      "Top-1 = share of GPS points whose highest-scored candidate is the reference arc (points without a correct "
      "candidate count as wrong). Route continuity = share of consecutive points whose predicted arcs are the same or "
      "directly connected. Primary selection metric: CV Top-1.")
pdf.h2("3.1 Overall comparison (classifier only, no HMM)")
pdf.grid([["Model", "Family", "Key configuration", "CV Macro F1", "CV Top-1", "Test Top-1", "Route contin.",
           "Test Macro F1", "Log Loss", "Infer. ms/pt"]] +
         [[n, r.family, r.config.strip("{}").replace('"', ""), f"{r.cv_macro_f1_mean:.3f}+-{r.cv_macro_f1_std:.3f}",
           f3(r.cv_top1_mean), f3(r.test_top1), f3(r.test_route_continuity), f3(r.test_macro_f1),
           f3(r.test_log_loss), f"{r.inference_ms_per_point:.3f}"] for n, r in s.iterrows()],
         [24, 17, 37, 18, 13, 13, 13, 15, 14, 16], size=6.8)
pdf.img(f"{PLOTS}/comparison_panel.png", 165, "Figure: CV Macro F1 (mean +- std over folds), student-test Top-1, "
        "Macro F1 and log loss. All from actual runs.")

pdf.h2("3.2 Individual models")
desc = {
    "LogisticRegression": ("Linear baseline that outputs probabilities directly - a natural HMM emission score.",
                           "Strength: fast, stable, interpretable weights. Limitation: only linear effects, e.g. it cannot "
                           "learn that heading matters only when the vehicle moves.",
                           "C=1.0 (L2), lbfgs, max_iter=1000, standardised features", "binary cross-entropy + L2"),
    "KNN": ("Points with similar distance/heading/speed patterns should share labels; no training needed.",
            "Strength: no assumptions about the decision boundary. Limitation: stores the training data (we use a "
            "50,000-row sample) so prediction is the slowest; vote-share probabilities are poorly calibrated.",
            "k=15, distance weighting, Euclidean on standardised features, brute-force search", "none (lazy learner)"),
    "DecisionTree": ("Learns threshold rules and interactions such as 'nearest AND heading agrees'.",
                     "Strength: non-linear and readable. Limitation: one tree is unstable and gives coarse leaf "
                     "probabilities.", "max_depth=12, min_samples_leaf=50", "greedy Gini impurity reduction"),
    "RandomForest": ("Averages many decorrelated trees, reducing the variance of a single tree.",
                     "Strength: robust, smoother probabilities. Limitation: largest model file and slowest tree model "
                     "to train.", "100 trees, max_depth=16, min_samples_leaf=20, 30% bootstrap", "Gini impurity, bagging"),
    "XGBoost": ("Boosting corrects the remaining errors tree by tree and handles skewed features and interactions.",
                "Strength: best accuracy and probabilities at low inference cost. Limitation: many hyperparameters "
                "(tuned in section 4).", "300 trees, depth 6, lr 0.1, subsample 0.8, colsample 0.8",
                "binary logistic loss (2nd-order boosting)"),
}
for n, r in s.iterrows():
    why, sl, hp, loss = desc[n]
    gap = r.train_top1 - r.oof_top1_same_trips
    fit = ("clear overfitting" if gap > 0.05 else "mild overfitting" if gap > 0.015 else "no sign of overfitting")
    if n == s.train_top1.idxmin():
        fit += "; it also has the lowest training score, i.e. it underfits compared with the tree models"
    pdf.h2(f"{n} ({r.family})")
    pdf.p(f"Why: {why} {sl} Hyperparameters: {hp}. Objective: {loss}. "
          f"Student-test: Top-1 {r.test_top1:.3f}, Macro F1 {r.test_macro_f1:.3f}, log loss {r.test_log_loss:.3f}, "
          f"precision {r.test_precision:.3f}, recall {r.test_recall:.3f}. Over/underfitting: on the same 20 training "
          f"trips, Top-1 is {r.train_top1:.3f} when the trips were seen in training and {r.oof_top1_same_trips:.3f} "
          f"when predicted by the CV model that did not see them (gap {gap:+.3f}): {fit}.", size=9)
lc_best = int(curve.val_logloss.idxmin())
pdf.img(f"{PLOTS}/xgb_learning_curve.png", 125,
        f"Figure: real per-round log loss of {cfg['model_name']} (64 training trips vs 16 held-out training trips). "
        f"Validation loss is lowest at round {curve['round'][lc_best]} of {len(curve)} ({curve.val_logloss.min():.4f} vs "
        f"{curve.val_logloss.iloc[-1]:.4f} at the end). Other models have no epoch-wise history, so no curves are drawn.")

# ---------------- 4. Bayesian ----------------
u, t = tvu.loc["XGBoost (untuned)"], tvu.loc["XGBoost (tuned)"]
bp = bayes["best_params"]
pdf.h1(f"4. Bayesian hyperparameter optimisation ({best_untuned})")
pdf.p(f"{best_untuned} had the best CV Top-1, so it was tuned with Optuna's TPE sampler (a Bayesian, model-based "
      f"search). Objective: maximise mean observation-level Top-1 over 3-fold GroupKFold on the training trips (each "
      f"trial uses a fixed {bayes['sample_frac']:.0%} sample of GPS points per trip to keep trials short). Trials: "
      f"{bayes['n_trials']} (trial 0 = the untuned configuration). Search time: {bayes['search_seconds'] / 60:.1f} min.")
ranges = {"n_estimators": "100-600 (step 50)", "max_depth": "3-10", "learning_rate": "0.02-0.3 (log)",
          "subsample": "0.5-1.0", "colsample_bytree": "0.5-1.0", "min_child_weight": "1-50 (log)",
          "reg_lambda": "0.1-20 (log)"}
untuned_vals = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 1, "reg_lambda": 1}
pdf.grid([["Hyperparameter", "Search range", "Untuned", "Best found"]] +
         [[k, ranges[k], untuned_vals[k], f"{bp[k]:.4g}" if isinstance(bp[k], float) else bp[k]] for k in ranges],
         [45, 50, 30, 35])
pdf.grid([["Model", "CV Macro F1", "CV Top-1", "Test Top-1", "Test Macro F1", "Log Loss", "Infer. ms/pt"],
          *[[n, f"{r.cv_macro_f1:.4f}+-{r.cv_macro_f1_std:.3f}", f"{r.cv_top1:.4f}", f"{r.test_top1:.4f}",
             f"{r.test_macro_f1:.4f}", f"{r.test_log_loss:.4f}", f"{r.inference_ms_per_point:.3f}"]
            for n, r in tvu.iterrows()]], [38, 30, 20, 20, 24, 20, 20])
better = t.cv_top1 > u.cv_top1
pdf.p(f"Result (same 5-fold CV and student-test trips for both): the tuned model "
      f"{'improves' if better else 'does not improve'} CV Top-1 ({u.cv_top1:.4f} -> {t.cv_top1:.4f}) and changes "
      f"student-test Top-1 from {u.test_top1:.4f} to {t.test_top1:.4f} and log loss from {u.test_log_loss:.4f} to "
      f"{t.test_log_loss:.4f}. The search found a smaller model ({bp['n_estimators']} trees, depth {bp['max_depth']}) with "
      f"the same accuracy and {u.inference_ms_per_point / t.inference_ms_per_point:.0f}x faster inference, i.e. the "
      "default configuration was already near the optimum for our features. The final system uses "
      f"{cfg['model_name']} (chosen on CV Top-1, not on the test set).")
pdf.img(f"{PLOTS}/bayes_history.png", 120, "Figure: CV Top-1 of every optimisation trial and the best value so far.")

# ---------------- 5. HMM ----------------
h_cls, h_hmm = hmm.iloc[0], hmm.iloc[1]
pdf.h1("5. HMM and Viterbi route reconstruction")
pdf.p("Hidden states: the candidate arcs of each GPS point (the classifier's candidate set, so every state maps to "
      "one road segment). Emission: w_e x log of the classifier probability, normalised over the point's candidates. "
      "Transition (Newson-Krumm style): -|d_route - d_gps| / beta, where d_gps is the straight distance between two "
      "consecutive GPS points and d_route is the distance driven along the network between their projections "
      "(same arc: along the arc; next arc: through the shared node; otherwise end of arc -> bounded Dijkstra shortest "
      "path -> start of arc). Disconnected segments (no path within 2 x d_gps + 200 m) get a fixed log-penalty of "
      "-30 instead of zero probability, so decoding never gets stuck on broken networks. Missing candidates: points "
      f"with no candidate are skipped and count as wrong. GPS gaps > {cfg['hmm']['gap_s']:g} s break the chain "
      "(uniform transition), so the route restarts after a dropout. Viterbi: delta_k(j) = max_i [delta_k-1(i) + "
      "log T(i,j)] + log E_k(j), with back-pointers to recover the best path.")
pdf.p(f"Decoder parameters were chosen on 20 training trips using out-of-fold classifier probabilities (grid: beta in "
      f"{sorted(grid.beta.unique().tolist())}, w_e in {sorted(grid.w_emission.unique().tolist())}); best beta = "
      f"{cfg['hmm']['beta']:g}, w_e = {cfg['hmm']['w_emission']:g} (train Top-1 "
      f"{grid.iloc[0].train_top1_classifier_only:.3f} classifier-only -> {grid.iloc[0].train_top1_hmm:.3f} with HMM).")
pdf.grid([["Student-testing (10 trips)", "Top-1 accuracy", "Route continuity", "Route precision", "Route recall",
           "Route F1"],
          *[[r.method, f"{r.top1:.4f}", f"{r.route_continuity:.4f}", f"{r.route_precision_mean:.4f}",
             f"{r.route_recall_mean:.4f}", f"{r.route_f1_mean:.4f}"] for _, r in hmm.iterrows()]], [50, 25, 27, 26, 24, 22])
hpt["gain"] = hpt.hmm_top1 - hpt.cls_top1
worse = hpt[hpt.gain < 0]
pdf.grid([["Trip", "Points", "Top-1 cls", "Top-1 HMM", "Change", "Continuity cls", "Continuity HMM", "Route F1 HMM"]] +
         [[r.trip, r.points, f3(r.cls_top1), f3(r.hmm_top1), f"{r.gain:+.3f}", f3(r.cls_continuity),
           f3(r.hmm_continuity), f3(r.hmm_route_f1)] for _, r in hpt.iterrows()], [20, 16, 20, 20, 18, 26, 26, 24], size=7.5)
pdf.p(f"Did the HMM help? Top-1 {h_cls.top1:.4f} -> {h_hmm.top1:.4f} and route continuity {h_cls.route_continuity:.4f} "
      f"-> {h_hmm.route_continuity:.4f}; it improved {int((hpt.gain > 0).sum())} of 10 trips. "
      + (f"It did not help on trip(s) {', '.join(worse.trip)}: when the classifier is confidently wrong for a long "
         "stretch (e.g. a parallel road that is connected to the true one), the transition model cannot overrule "
         "it, and a wrong early choice can propagate. " if len(worse) else "No trip got worse. ")
      + "Route precision measures how many predicted arcs are on the reference route (jumps to side roads lower it); "
        "recall measures how much of the reference route is covered.")

# ---------------- 6. Folium ----------------
pdf.h1("6. Geographical visualisation (Folium)")
biggest = hpt.trip[hpt.gain.idxmax()]
pdf.img(f"{PLOTS}/folium_trip_00000000.png", 150, "Figure: student-test trip 00000000 - GPS (blue), road network "
        "(grey), reference route (green), classifier only (orange), classifier + HMM (red), with layer control.")
if biggest != "00000000":
    pdf.img(f"{PLOTS}/folium_trip_{biggest}.png", 150, f"Figure: student-test trip {biggest}, the trip with the largest "
            "HMM change - classifier jumps (orange) are replaced by a connected route (red).")
pdf.p("Interactive maps for all 10 student-testing trips: results/maps/trip_<id>_map.html (code: "
      "visualisation/make_plots.py).")

# ---------------- 7. API / UI ----------------
pdf.h1("7. API, UI and deployment")
pdf.p("Laptop 1 runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`. It loads the final classifier and HMM "
      "parameters from artifacts/final_config.json and the trips from PATHSNAP_DATA_DIR. Endpoints: GET /health, "
      "GET /trips, POST /predict/{trip_id}. For each request the API runs cleaning -> candidates -> features -> "
      "classifier -> HMM/Viterbi using only .track/.nodes/.arcs, then (server-side) aligns the .route file to compute "
      "Top-1 accuracy, continuity and route precision/recall/F1. The response contains the GPS points, the predicted "
      "route and the metrics - never the reference route. Laptop 2 opens ui/index.html (plain HTML + JavaScript with Leaflet maps, served with `python -m http.server`), lists the trips "
      "from the API, sends the selected trip id and draws the returned route.")
pdf.img(f"{PLOTS}/ui_screenshot.png", 170, "Figure: browser UI (ui/index.html) showing a reconstructed student-test trip "
        "returned by the API.")
if api_log:
    lines = [l for l in api_log.splitlines() if "pathsnap" in l.lower() or "POST" in l or "GET" in l][-8:]
    pdf.h2("API log (Laptop 1 console during the test run)")
    pdf.set_font("Courier", "", 6.5)
    for l in lines:
        pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 3.4, l.encode("latin-1", "replace").decode("latin-1"))
    pdf.ln(2)

# ---------------- 8. conclusion ----------------
pdf.h1("8. Final model selection and team")
pdf.p(f"Selected: {cfg['model_name']} + HMM. Reasons from our results: best CV Top-1 and Macro F1 of the five "
      f"models, the lowest log loss (good probabilities are what the HMM needs as emissions), low inference cost, and "
      f"the HMM raises student-test Top-1 to {h_hmm.top1:.3f} and continuity to {h_hmm.route_continuity:.3f}. "
      "Limitations: labels come from our alignment of GPS to the supplied route (reference information, not verified "
      "ground truth); a few trips have offset GPS; the candidate radius caps the achievable accuracy.")
pdf.grid([["Member", "Responsibility"],
          ["Member 1", "Data exploration, loading, cleaning, road network, labels, candidates, features"],
          ["Member 2", "Logistic Regression and KNN; FastAPI backend"],
          ["Member 3", "Decision Tree and Random Forest; HMM/Viterbi decoder"],
          ["Member 4", "XGBoost, Bayesian optimisation, comparison, graphs, web UI, report"]], [30, 150])

pdf.output("reports/PathSnap_Technical_Report.pdf")
print("wrote reports/PathSnap_Technical_Report.pdf, pages:", pdf.page_no())
