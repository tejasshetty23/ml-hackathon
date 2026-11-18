"""EDA plots, model comparison graphs, confusion matrices and a Folium map.

Run:  py -3.13 -m visualisation.make_plots      (from the PathSnap folder, after training)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from preprocessing.pipeline import load_trip, RAW

PLOTS, MAPS, MET = "results/plots", "results/maps", "results/metrics"
os.makedirs(PLOTS, exist_ok=True); os.makedirs(MAPS, exist_ok=True)
MAP_TRIP = "00000000"  # student-testing trip used for the trajectory plot and the Folium map


def dataset_summary():
    rows = []
    for part in ["train", "student_test"]:
        for d in sorted(glob.glob(f"{RAW}/{part}/*")):
            tid, tr, nd, ar, rt = load_trip(d)
            dt = np.diff(tr.t.values)
            rows.append({"partition": part, "trip": tid, "gps_points": len(tr), "nodes": len(nd), "arcs": len(ar),
                         "route_arcs": len(rt), "missing_values": int(tr.isna().sum().sum() + nd.isna().sum().sum()
                                                                      + ar.isna().sum().sum()),
                         "share_dt_1s": (dt == 1).mean(), "max_gap_s": dt.max(), "gaps_over_5s": int((dt > 5).sum()),
                         "duration_s": tr.t.iloc[-1] - tr.t.iloc[0]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{MET}/dataset_inventory.csv", index=False)
    s = df.groupby("partition").agg(trips=("trip", "count"), gps_points=("gps_points", "sum"),
                                    median_points=("gps_points", "median"), median_arcs=("arcs", "median"),
                                    missing_values=("missing_values", "sum"),
                                    trips_with_gap_over_60s=("max_gap_s", lambda x: int((x > 60).sum())))
    s.to_csv(f"{MET}/dataset_summary.csv")
    print(s)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].hist(df[df.partition == "train"].gps_points, bins=30, color="#4C72B0")
    ax[0].set(title="GPS points per trip (train)", xlabel="points", ylabel="trips")
    allgaps = np.concatenate([np.diff(load_trip(d)[1].t.values) for d in sorted(glob.glob(f"{RAW}/train/*"))])
    ax[1].hist(np.clip(allgaps, 0, 30), bins=30, color="#4C72B0", log=True)
    ax[1].set(title="Time between GPS points (train, clipped at 30 s)", xlabel="seconds", ylabel="count (log)")
    pts = pd.read_parquet("data/processed/train_points_all.parquet")  # before label cleaning
    ax[2].hist(np.clip(pts.label_dist, 0, 100), bins=50, color="#4C72B0", log=True)
    ax[2].set(title="GPS distance to reference road (train)", xlabel="metres (clipped at 100)", ylabel="points (log)")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/eda_distributions.png", dpi=130); plt.close()

    miss = pts.groupby("trip").label_in_cands.mean().rsub(1).sort_values(ascending=False)
    plt.figure(figsize=(14, 3.2)); plt.bar(miss.index.str[-2:], miss.values, color="#C44E52")
    plt.title("Candidate miss rate per training trip (reference road not within 50 m / top-8)")
    plt.xlabel("trip id (last 2 digits)"); plt.ylabel("miss rate"); plt.xticks(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/eda_candidate_miss_per_trip.png", dpi=130); plt.close()


def trajectory_plot_and_map(trip=MAP_TRIP):
    tid, tr, nd, ar, rt = load_trip(f"{RAW}/student_test/{trip}")
    pad = 0.003
    box = ((nd.lon >= tr.lon.min() - pad) & (nd.lon <= tr.lon.max() + pad) &
           (nd.lat >= tr.lat.min() - pad) & (nd.lat <= tr.lat.max() + pad)).values
    near = ar[box[ar.u.values] & box[ar.v.values]]
    def pred_arcs(file):
        pred = pd.read_csv(f"{MET}/{file}", dtype={"trip": str})
        pred = pred[pred.trip == tid].sort_values("point")
        return pred.arc.values[np.r_[True, pred.arc.values[1:] != pred.arc.values[:-1]]]
    cls_arcs = pred_arcs("student_test_predictions_final_classifier.csv")
    hmm_arcs = pred_arcs("student_test_predictions_hmm.csv")

    # static plot
    fig, ax = plt.subplots(figsize=(8, 8))
    seg = lambda a: (nd.lon.values[ar.u.values[a]], nd.lon.values[ar.v.values[a]],
                     nd.lat.values[ar.u.values[a]], nd.lat.values[ar.v.values[a]])
    x0, x1, y0, y1 = seg(near.index.values)
    ax.plot(np.c_[x0, x1].T, np.c_[y0, y1].T, color="#BBBBBB", lw=0.5)
    x0, x1, y0, y1 = seg(rt)
    ax.plot(np.c_[x0, x1].T, np.c_[y0, y1].T, color="green", lw=3)
    ax.scatter(tr.lon, tr.lat, s=3, color="blue", zorder=3)
    ax.set_title(f"Student-test trip {tid}: road network (grey), reference route (green), GPS (blue)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/trajectory_{tid}.png", dpi=130); plt.close()

    # Folium map with layers
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    layers = {k: folium.FeatureGroup(name=k) for k in
              ["Road network (nearby)", "GPS observations", "Reference route", "Classifier only", "Classifier + HMM"]}  # later layers are drawn on top
    for a in near.index.values[:6000]:
        folium.PolyLine([(nd.lat[ar.u[a]], nd.lon[ar.u[a]]), (nd.lat[ar.v[a]], nd.lon[ar.v[a]])],
                        color="#999999", weight=1).add_to(layers["Road network (nearby)"])
    for arcs, name, col, w in [(rt, "Reference route", "green", 8), (cls_arcs, "Classifier only", "orange", 4),
                               (hmm_arcs, "Classifier + HMM", "red", 3)]:
        for a in arcs:
            folium.PolyLine([(nd.lat[ar.u[a]], nd.lon[ar.u[a]]), (nd.lat[ar.v[a]], nd.lon[ar.v[a]])],
                            color=col, weight=w, opacity=0.8).add_to(layers[name])
    for lat, lon, t in tr[["lat", "lon", "t"]].values[::2]:
        folium.CircleMarker((lat, lon), radius=1.5, color="blue", fill=True, opacity=0.5, popup=f"t={t:.0f}s").add_to(layers["GPS observations"])
    for g in layers.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:999;background:white;padding:8px;border:1px solid #999;'
        f'font-size:13px"><b>Trip {tid} (student-test)</b><br><span style="color:blue">&#9679;</span> GPS observations<br>'
        '<span style="color:green">&#9644;</span> Reference route<br><span style="color:orange">&#9644;</span> Classifier only'
        '<br><span style="color:red">&#9644;</span> Classifier + HMM'
        '<br><span style="color:#999">&#9644;</span> Road network</div>'))
    m.save(f"{MAPS}/trip_{tid}_map.html")


def best_model():
    # primary selection metric: cross-validated observation-level Top-1 accuracy (training trips only)
    return pd.read_csv(f"{MET}/model_comparison.csv").sort_values("cv_top1_mean").model.iloc[-1]


def comparison_plots():
    s = pd.read_csv(f"{MET}/model_comparison.csv")
    cv = pd.read_csv(f"{MET}/cv_results.csv")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    specs = [("cv_macro_f1_mean", "cv_macro_f1_std", "Cross-validation Macro F1 (5-fold GroupKFold, train)", "cv_macro_f1.png"),
             ("test_top1", None, "Student-testing observation-level Top-1 accuracy", "test_top1.png"),
             ("test_macro_f1", None, "Student-testing Macro F1", "test_macro_f1.png"),
             ("test_log_loss", None, "Student-testing Log Loss (lower is better)", "test_log_loss.png")]
    for col, err, title, fn in specs:
        plt.figure(figsize=(7, 3.8))
        b = plt.bar(s.model, s[col], yerr=s[err] if err else None, capsize=4, color=colors)
        plt.bar_label(b, fmt="%.4f", fontsize=8)
        lo = s[col].min()
        if "log_loss" not in col:
            plt.ylim(max(0, lo - 0.1), min(1.0, s[col].max() + 0.05))
        plt.title(title); plt.xticks(rotation=15); plt.tight_layout()
        plt.savefig(f"{PLOTS}/{fn}", dpi=130); plt.close()

    # the same four charts in one 2x2 panel (used in the PDF)
    short = s.model.replace({"LogisticRegression": "LogReg", "DecisionTree": "DTree", "RandomForest": "RForest"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (col, err, title, _) in zip(axes.ravel(), specs):
        b = ax.bar(short, s[col], yerr=s[err] if err else None, capsize=3, color=colors)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
        if "log_loss" not in col:
            ax.set_ylim(max(0, s[col].min() - 0.1), min(1.0, s[col].max() + 0.05))
        ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/comparison_panel.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 3.8))
    for i, (name, g) in enumerate(cv.groupby("model", sort=False)):
        plt.plot(g.fold, g.macro_f1, "o-", label=name, color=colors[i])
    plt.xlabel("fold"); plt.ylabel("Macro F1"); plt.title("Macro F1 per CV fold"); plt.legend(fontsize=8)
    plt.xticks(range(5)); plt.tight_layout(); plt.savefig(f"{PLOTS}/cv_folds.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, name in zip(axes, s.model):
        cm = np.load(f"{MET}/confusion_{name}.npy")
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, f"{val}\n({val / cm[r].sum():.2f})", ha="center", va="center", fontsize=8)
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["not road", "road"],
               yticklabels=["not road", "road"], xlabel="predicted", ylabel="actual")
    plt.suptitle("Candidate-level confusion matrices, student-testing set (threshold 0.5)")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/confusion_matrices.png", dpi=130); plt.close()


def screenshot_map(trip=MAP_TRIP):
    """Static PNG of the Folium map for the PDF (uses a headless Edge/Chrome if available)."""
    import subprocess, shutil
    html = os.path.abspath(f"{MAPS}/trip_{trip}_map.html")
    png = os.path.abspath(f"{PLOTS}/folium_trip_{trip}.png")
    for exe in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if os.path.exists(exe):
            subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=8000",
                            f"--screenshot={png}", "--window-size=1200,900", "file:///" + html.replace("\\", "/")],
                           timeout=90, capture_output=True)
            if os.path.exists(png):
                print("map screenshot ->", png); return
    print("no browser screenshot available")


def bayes_plot():
    t = pd.read_csv(f"{MET}/bayes_trials.csv")
    plt.figure(figsize=(7, 3.4))
    plt.plot(t.number, t.value, "o", color="#4C72B0", label="trial (3-fold CV Top-1)")
    plt.plot(t.number, t.value.cummax(), "-", color="#C44E52", label="best so far")
    plt.scatter([0], [t.value.iloc[0]], s=80, facecolors="none", edgecolors="black", label="trial 0 = untuned config")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.title("Bayesian optimisation (Optuna TPE) of XGBoost")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{PLOTS}/bayes_history.png", dpi=130); plt.close()


if __name__ == "__main__":
    dataset_summary()
    comparison_plots()
    bayes_plot()
    for d in sorted(glob.glob(f"{RAW}/student_test/*")):  # one Folium map per student-testing trip
        trajectory_plot_and_map(os.path.basename(d))
    gains = pd.read_csv(f"{MET}/hmm_per_trip.csv", dtype={"trip": str})
    biggest = gains.trip[(gains.hmm_top1 - gains.cls_top1).idxmax()]
    for trip in {MAP_TRIP, biggest}:
        screenshot_map(trip)
    print("plots ->", PLOTS, " maps ->", MAPS, " largest HMM change:", biggest)
