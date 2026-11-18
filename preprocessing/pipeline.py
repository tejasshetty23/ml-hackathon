"""PathSnap data pipeline: load trip -> label GPS points -> candidate roads -> features.

Run:  py -3.13 -m preprocessing.pipeline      (from the PathSnap folder)
Output: data/processed/{train,student_test}.parquet  (one row per GPS point x candidate road)
"""
import os, glob, time
import numpy as np
import pandas as pd
import shapely

RAW = "data/raw"
OUT = "data/processed"
RADIUS_M = 50      # candidate search radius in metres
MAX_CANDS = 8      # keep the 8 nearest roads per GPS point
EARTH_R = 6371008.8

FEATURES = ["dist", "dist_rank", "dist_minus_best", "heading_diff", "heading_valid",
            "speed", "dt_prev", "dt_next", "arc_len", "proj_frac", "n_cands", "two_way"]


# ---------- 1. loading ----------
def load_trip(trip_dir):
    tid = os.path.basename(trip_dir)
    read = lambda ext, cols: pd.read_csv(f"{trip_dir}/{tid}.{ext}", sep=r"\s+", header=None, names=cols)
    track = read("track", ["lon", "lat", "t"])
    nodes = read("nodes", ["lon", "lat"])
    arcs = read("arcs", ["u", "v"])        # row index = arc_id
    route = read("route", ["arc"]).arc.values  # ordered arc_ids driven
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


# ---------- 3. labels: align each GPS point to the reference route ----------
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


# ---------- 4+5. candidates and features ----------
def process_trip(trip_dir):
    tid, track, nodes, arcs, route = load_trip(trip_dir)
    lat0 = track.lat.mean()
    nx, ny = to_metres(nodes.lon.values, nodes.lat.values, lat0)
    px, py = to_metres(track.lon.values, track.lat.values, lat0)
    t = track.t.values.astype(float)
    u, v = arcs.u.values, arcs.v.values

    # labels
    pos = align_to_route(px, py, nx[u[route]], ny[u[route]], nx[v[route]], ny[v[route]])
    label_arc = route[pos]
    label_dist = point_to_segment(px, py, nx[u[label_arc]], ny[u[label_arc]], nx[v[label_arc]], ny[v[label_arc]])

    # GPS motion (central differences)
    ip, inx = np.r_[0, np.arange(len(px) - 1)], np.r_[np.arange(1, len(px)), len(px) - 1]
    disp = np.hypot(px[inx] - px[ip], py[inx] - py[ip])
    dtc = np.maximum(t[inx] - t[ip], 1)
    speed = disp / dtc
    gps_heading = np.degrees(np.arctan2(px[inx] - px[ip], py[inx] - py[ip]))  # 0 = north
    heading_valid = (disp > 2.0).astype(int)  # heading is meaningless when standing still
    dt_prev = np.r_[0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0]

    # candidate roads: all arcs within RADIUS_M (spatial index), keep MAX_CANDS nearest
    lines = shapely.linestrings(np.stack([np.c_[nx[u], ny[u]], np.c_[nx[v], ny[v]]], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(px, py)
    i, j = tree.query(pts, predicate="dwithin", distance=RADIUS_M)
    dist = shapely.distance(pts[i], lines[j])
    order = np.lexsort((dist, i))
    i, j, dist = i[order], j[order], dist[order]
    rank = pd.Series(i).groupby(i).cumcount().values
    keep = rank < MAX_CANDS
    i, j, dist, rank = i[keep], j[keep], dist[keep], rank[keep]

    arc_heading = np.degrees(np.arctan2(nx[v] - nx[u], ny[v] - ny[u]))
    arc_len = np.hypot(nx[v] - nx[u], ny[v] - ny[u])
    pair_set = set(zip(u.tolist(), v.tolist()))
    two_way = np.fromiter(((b, a) in pair_set for a, b in zip(u[j], v[j])), bool, len(j))
    n_cands = np.bincount(i, minlength=len(px))
    best = np.full(len(px), np.nan); best[i[rank == 0]] = dist[rank == 0]
    hd = angle_diff_deg(gps_heading[i], arc_heading[j])
    hd = np.where(heading_valid[i] == 1, hd, 90.0)

    # a candidate is correct if it is the reference arc (or an exact duplicate arc u->v)
    y = ((u[j] == u[label_arc[i]]) & (v[j] == v[label_arc[i]])).astype(int)

    df = pd.DataFrame({
        "trip": tid, "point": i, "arc": j, "u": u[j], "v": v[j], "label_arc": label_arc[i], "y": y,
        "dist": dist, "dist_rank": rank, "dist_minus_best": dist - best[i], "heading_diff": hd,
        "heading_valid": heading_valid[i], "speed": speed[i], "dt_prev": dt_prev[i], "dt_next": dt_next[i],
        "arc_len": arc_len[j], "proj_frac": shapely.line_locate_point(lines[j], pts[i], normalized=True),
        "n_cands": n_cands[i], "two_way": two_way.astype(int),
    })
    points = pd.DataFrame({"trip": tid, "point": np.arange(len(px)), "label_arc": label_arc,
                           "label_dist": label_dist, "n_cands": n_cands,
                           "label_in_cands": np.bincount(i, weights=y, minlength=len(px)) > 0})
    return df, points


def build(partition):
    frames, pts = [], []
    for d in sorted(glob.glob(f"{RAW}/{partition}/*")):
        t0 = time.time()
        df, p = process_trip(d)
        frames.append(df); pts.append(p)
        print(f"  {partition}/{os.path.basename(d)}: {len(p)} pts, {len(df)} cand rows, "
              f"miss={1 - p.label_in_cands.mean():.3f}  ({time.time() - t0:.1f}s)", flush=True)
    df, pts = pd.concat(frames, ignore_index=True), pd.concat(pts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/{partition}.parquet")
    pts.to_parquet(f"{OUT}/{partition}_points.parquet")
    print(f"{partition}: {pts.trip.nunique()} trips, {len(pts)} points, {len(df)} rows, "
          f"candidate miss rate = {1 - pts.label_in_cands.mean():.4f}")


if __name__ == "__main__":
    for part in ["train", "student_test"]:
        build(part)
