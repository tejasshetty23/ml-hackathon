"""PathSnap data pipeline: load trip -> clean -> candidate roads -> features (+ labels for training).

Run:  py -3.13 -m preprocessing.pipeline      (from the PathSnap folder)
Output: data/processed/{train,student_test}.parquet  (one row per GPS point x candidate road)
        data/processed/{train,student_test}_points.parquet (one row per GPS point)
        results/metrics/cleaning_report.csv

Inference (API) uses only `candidates_and_features` - it never sees the .route file.
"""
import os, glob, time
import numpy as np
import pandas as pd
import shapely

RAW = "data/raw"
OUT = "data/processed"
RADIUS_M = 50            # candidate search radius in metres
MAX_CANDS = 8            # keep the 8 nearest roads per GPS point
MIN_ARC_LEN_M = 0.5      # cleaning: arcs shorter than this (or self-loops) are not real road segments
MAX_LABEL_DIST_M = 100   # cleaning (training only): labels further than this from the GPS point are unreliable
EARTH_R = 6371008.8

FEATURES = ["dist", "dist_rank", "dist_minus_best", "heading_diff", "heading_valid",
            "speed", "dt_prev", "dt_next", "arc_len", "proj_frac", "n_cands", "two_way"]


# ---------- 1. loading ----------
def read_trip_files(trip_dir):
    tid = os.path.basename(trip_dir)
    read = lambda ext, cols: pd.read_csv(f"{trip_dir}/{tid}.{ext}", sep=r"\s+", header=None, names=cols)
    track = read("track", ["lon", "lat", "t"])
    nodes = read("nodes", ["lon", "lat"])
    arcs = read("arcs", ["u", "v"])        # row index = arc_id
    route = read("route", ["arc"]).arc.values if os.path.exists(f"{trip_dir}/{tid}.route") else None
    return tid, track, nodes, arcs, route


def clean_track(track):
    """Cleaning of GPS rows: drop rows with missing/invalid coordinates, non-increasing or duplicate timestamps.
    (In the supplied data none are found, but the API must be safe on any trip.)"""
    ok = track[["lon", "lat", "t"]].notna().all(axis=1)
    ok &= track.lon.between(-180, 180) & track.lat.between(-90, 90)
    track = track[ok]
    track = track[track.t.diff().fillna(1) > 0]   # keep strictly increasing time
    return track.reset_index(drop=True), int((~ok).sum())


def load_trip(trip_dir):
    tid, track, nodes, arcs, route = read_trip_files(trip_dir)
    track, _ = clean_track(track)
    return tid, track, nodes, arcs, route


# ---------- 2. geography ----------
def to_metres(lon, lat, lat0):
    """Local equirectangular projection (metres). Accurate at city scale; each trip uses its own lat0."""
    x = np.radians(lon) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_R
    return x, y


def point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    s = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    return np.hypot(px - (ax + s * dx), py - (ay + s * dy))


def angle_diff_deg(a, b):
    return np.abs((a - b + 180) % 360 - 180)


# ---------- 3. candidates and features (no labels - used for training AND inference) ----------
def candidates_and_features(tid, track, nodes, arcs):
    lat0 = track.lat.mean()
    nx, ny = to_metres(nodes.lon.values, nodes.lat.values, lat0)
    px, py = to_metres(track.lon.values, track.lat.values, lat0)
    t = track.t.values.astype(float)
    u, v = arcs.u.values, arcs.v.values
    arc_len = np.hypot(nx[v] - nx[u], ny[v] - ny[u])

    # GPS motion (central differences)
    ip, inx = np.r_[0, np.arange(len(px) - 1)], np.r_[np.arange(1, len(px)), len(px) - 1]
    disp = np.hypot(px[inx] - px[ip], py[inx] - py[ip])
    speed = disp / np.maximum(t[inx] - t[ip], 1)
    gps_heading = np.degrees(np.arctan2(px[inx] - px[ip], py[inx] - py[ip]))  # 0 = north
    heading_valid = (disp > 2.0).astype(int)  # heading is meaningless when standing still
    dt_prev = np.r_[0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0]

    # cleaning of the road network: self-loops and zero-length arcs are never candidates
    valid_arcs = np.flatnonzero((u != v) & (arc_len >= MIN_ARC_LEN_M))
    lines = shapely.linestrings(np.stack([np.c_[nx[u], ny[u]], np.c_[nx[v], ny[v]]], axis=1))
    tree = shapely.STRtree(lines[valid_arcs])
    pts = shapely.points(px, py)
    i, jj = tree.query(pts, predicate="dwithin", distance=RADIUS_M)
    j = valid_arcs[jj]
    dist = shapely.distance(pts[i], lines[j])
    order = np.lexsort((dist, i))
    i, j, dist = i[order], j[order], dist[order]
    rank = pd.Series(i).groupby(i).cumcount().values
    keep = rank < MAX_CANDS
    i, j, dist, rank = i[keep], j[keep], dist[keep], rank[keep]

    arc_heading = np.degrees(np.arctan2(nx[v] - nx[u], ny[v] - ny[u]))
    pair_set = set(zip(u.tolist(), v.tolist()))
    two_way = np.fromiter(((b, a) in pair_set for a, b in zip(u[j], v[j])), bool, len(j))
    n_cands = np.bincount(i, minlength=len(px))
    best = np.full(len(px), np.nan); best[i[rank == 0]] = dist[rank == 0]
    hd = np.where(heading_valid[i] == 1, angle_diff_deg(gps_heading[i], arc_heading[j]), 90.0)

    df = pd.DataFrame({
        "trip": tid, "point": i, "arc": j, "u": u[j], "v": v[j],
        "dist": dist, "dist_rank": rank, "dist_minus_best": dist - best[i], "heading_diff": hd,
        "heading_valid": heading_valid[i], "speed": speed[i], "dt_prev": dt_prev[i], "dt_next": dt_next[i],
        "arc_len": arc_len[j], "proj_frac": shapely.line_locate_point(lines[j], pts[i], normalized=True),
        "n_cands": n_cands[i], "two_way": two_way.astype(int),
    })
    geo = {"px": px, "py": py, "t": t, "n_cands": n_cands, "nx": nx, "ny": ny, "arc_len": arc_len,
           "n_invalid_arcs": int(len(u) - len(valid_arcs))}
    return df, geo


# ---------- 4. labels: align each GPS point to the reference route (training/scoring only) ----------
def align_to_route(px, py, ax, ay, bx, by, cap=150.0):
    """Monotone alignment: GPS point n gets route position m(n), with m never going backwards
    (the vehicle drives the route in order). Minimises total point-to-arc distance (capped).
    Simple dynamic programming: C[n,m] = d[n,m] + min_{m'<=m} C[n-1,m']."""
    N, M = len(px), len(ax)
    back = np.zeros((N, M), dtype=np.int32)
    ar = np.arange(M)
    prev = None
    for n in range(N):
        d = np.minimum(point_to_segment(px[n], py[n], ax, ay, bx, by), cap)
        if prev is None:
            prev = d
            continue
        best = np.minimum.accumulate(prev)
        back[n] = np.maximum.accumulate(np.where(prev <= best, ar, 0))  # argmin of prev[0..m]
        prev = d + best
    path = np.empty(N, dtype=np.int64)
    path[-1] = int(np.argmin(prev))
    for n in range(N - 1, 0, -1):
        path[n - 1] = back[n, path[n]]
    return path


def route_labels(geo, arcs, route):
    """Reference arc for every GPS point + its distance to the point."""
    px, py, nx, ny = geo["px"], geo["py"], geo["nx"], geo["ny"]
    u, v = arcs.u.values, arcs.v.values
    pos = align_to_route(px, py, nx[u[route]], ny[u[route]], nx[v[route]], ny[v[route]])
    label_arc = route[pos]
    label_dist = point_to_segment(px, py, nx[u[label_arc]], ny[u[label_arc]], nx[v[label_arc]], ny[v[label_arc]])
    return label_arc, label_dist


def add_labels(df, label_arc, arcs):
    """A candidate is correct if it is the reference arc (or an exact duplicate arc u->v)."""
    u, v = arcs.u.values, arcs.v.values
    la = label_arc[df.point.values]
    df["label_arc"] = la
    df["y"] = ((df.u.values == u[la]) & (df.v.values == v[la])).astype(int)
    return df


def process_trip(trip_dir):
    tid, track, nodes, arcs, route = read_trip_files(trip_dir)
    track, n_bad = clean_track(track)
    df, geo = candidates_and_features(tid, track, nodes, arcs)
    label_arc, label_dist = route_labels(geo, arcs, route)
    df = add_labels(df, label_arc, arcs)
    n = len(track)
    points = pd.DataFrame({"trip": tid, "point": np.arange(n), "label_arc": label_arc, "label_dist": label_dist,
                           "n_cands": geo["n_cands"],
                           "label_in_cands": np.bincount(df.point.values, weights=df.y.values, minlength=n) > 0})
    clean = {"trip": tid, "gps_rows_removed": n_bad, "arcs_excluded": geo["n_invalid_arcs"]}
    return df, points, clean


def build(partition):
    frames, pts, cleans = [], [], []
    for d in sorted(glob.glob(f"{RAW}/{partition}/*")):
        t0 = time.time()
        df, p, c = process_trip(d)
        frames.append(df); pts.append(p); cleans.append(c)
        print(f"  {partition}/{os.path.basename(d)}: {len(p)} pts, {len(df)} cand rows, "
              f"miss={1 - p.label_in_cands.mean():.3f}  ({time.time() - t0:.1f}s)", flush=True)
    df, pts, cl = pd.concat(frames, ignore_index=True), pd.concat(pts, ignore_index=True), pd.DataFrame(cleans)
    cl["partition"] = partition
    os.makedirs(OUT, exist_ok=True)
    pts.to_parquet(f"{OUT}/{partition}_points_all.parquet")  # before label cleaning (used for EDA)

    if partition == "train":
        # cleaning (training only): drop GPS points whose reference road is > MAX_LABEL_DIST_M away - their label
        # is unreliable (offset / very noisy trips). The student-testing partition is never filtered.
        bad = pts.label_dist > MAX_LABEL_DIST_M
        cl = cl.merge(pts[bad].groupby("trip").size().rename("points_unreliable_label").reset_index(),
                      on="trip", how="left").fillna({"points_unreliable_label": 0})
        bad_keys = set(zip(pts.trip[bad], pts.point[bad]))
        df = df[[k not in bad_keys for k in zip(df.trip, df.point)]]
        pts = pts[~bad]
        print(f"cleaning: removed {bad.sum()} training points ({bad.mean():.2%}) with label distance > "
              f"{MAX_LABEL_DIST_M} m")
    else:
        cl["points_unreliable_label"] = 0
    df.reset_index(drop=True).to_parquet(f"{OUT}/{partition}.parquet")
    pts.reset_index(drop=True).to_parquet(f"{OUT}/{partition}_points.parquet")
    print(f"{partition}: {pts.trip.nunique()} trips, {len(pts)} points, {len(df)} rows, "
          f"candidate miss rate = {1 - pts.label_in_cands.mean():.4f}")
    return cl


if __name__ == "__main__":
    os.makedirs("results/metrics", exist_ok=True)
    pd.concat([build(p) for p in ["train", "student_test"]]).to_csv("results/metrics/cleaning_report.csv", index=False)
