"""PathSnap FastAPI backend (Laptop 1).

Run (from the PathSnap folder):
    set PATHSNAP_DATA_DIR=data/evaluator_test     (folder with one sub-folder per trip; default: data/raw/student_test)
    py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:  GET /health   GET /trips   POST /predict/{trip_id}
Prediction uses only .track/.nodes/.arcs. The .route file (ground truth) is read only AFTER prediction, server-side,
to compute accuracy; it is never returned to the UI.
"""
import os, json, time, logging
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from preprocessing.pipeline import read_trip_files, clean_track, candidates_and_features, route_labels, FEATURES
from models.hmm import RoadGraph, prepare_trip, viterbi
from training.train_models import route_continuity
import pandas as pd

DATA_DIR = os.environ.get("PATHSNAP_DATA_DIR", "data/raw/student_test")
CFG = json.load(open("artifacts/final_config.json"))
MODEL = joblib.load(CFG["model_file"])
HMM = CFG["hmm"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")
log.info(f"model={CFG['model_name']}  hmm={HMM}  data_dir={DATA_DIR}")

app = FastAPI(title="PathSnap API", version="1.0")
# the browser UI (ui/index.html) runs on Laptop 2, i.e. a different origin -> allow cross-origin GET/POST
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def list_trips():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(d for d in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, d, f"{d}.track")))


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG["model_name"], "hmm": HMM, "trips": len(list_trips())}


@app.get("/trips")
def trips():
    t = list_trips()
    log.info(f"GET /trips -> {len(t)} trips")
    return {"trips": t}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trips():  # also blocks path tricks like ../
        raise HTTPException(404, f"unknown trip {trip_id}")
    log.info(f"POST /predict/{trip_id} - request received")
    t0 = time.time()

    # ---------- prediction pipeline (no ground truth involved) ----------
    tid, track, nodes, arcs, route = read_trip_files(os.path.join(DATA_DIR, trip_id))
    track, n_bad = clean_track(track)
    df, geo = candidates_and_features(tid, track, nodes, arcs)
    df["p"] = MODEL.predict_proba(df[FEATURES].values)[:, 1]
    u, v = arcs.u.values, arcs.v.values
    graph = RoadGraph(u, v, geo["arc_len"], len(nodes))
    hmm_pred = viterbi(prepare_trip(df, geo["px"], geo["py"], geo["t"], graph, HMM["gap_s"]),
                       HMM["beta"], HMM["w_emission"])
    cls_pred = df.loc[df.groupby("point").p.idxmax()].set_index("point").arc.to_dict()
    infer_s = time.time() - t0

    pts = np.array(sorted(hmm_pred), dtype=int)
    seq = np.array([hmm_pred[p] for p in pts], dtype=int)
    route_arcs = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    lat, lon = nodes.lat.values, nodes.lon.values
    response = {
        "trip_id": trip_id, "model": CFG["model_name"], "gps_points": len(track), "invalid_gps_rows_removed": n_bad,
        "points_with_prediction": int(len(pts)),
        "gps": track[["lat", "lon"]].round(6).values.tolist(),
        "predicted_route": [[[lat[u[a]], lon[u[a]]], [lat[v[a]], lon[v[a]]]] for a in route_arcs],
        "inference_seconds": round(infer_s, 3),
    }

    # ---------- server-side evaluation (reference route stays on the server) ----------
    if route is not None:
        label_arc, _ = route_labels(geo, arcs, route)
        n = len(track)
        def acc(pred):
            a = np.array([pred.get(i, -1) for i in range(n)])
            ok = (a >= 0) & (u[np.maximum(a, 0)] == u[label_arc]) & (v[np.maximum(a, 0)] == v[label_arc])
            return float(ok.mean())
        P, R = set(route_arcs.tolist()), set(route.tolist())
        prec = len(P & R) / max(len(P), 1); rec = len(P & R) / len(R)
        pred_df = pd.DataFrame({"trip": trip_id, "point": pts, "arc": seq, "u": u[seq], "v": v[seq]})
        response["evaluation"] = {
            "top1_accuracy_hmm": round(acc(hmm_pred), 4),
            "top1_accuracy_classifier_only": round(acc(cls_pred), 4),
            "route_continuity": round(float(route_continuity(pred_df)), 4),
            "route_precision": round(prec, 4), "route_recall": round(rec, 4),
            "route_f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4),
        }
    response["total_seconds"] = round(time.time() - t0, 3)
    log.info(f"POST /predict/{trip_id} - {len(track)} points, inference {infer_s:.2f}s, "
             f"evaluation={response.get('evaluation')}")
    return response
