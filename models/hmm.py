"""HMM + Viterbi route reconstruction on top of the classifier's candidate probabilities.

Hidden state at GPS point k  = one of its candidate arcs (from the candidate generator).
Emission  log P(obs_k | arc)  = w_e * log( p_classifier(arc) / sum of p over the point's candidates )
Transition log P(arc_b | arc_a) = -|d_route(a->b) - d_gps(k-1->k)| / beta      (Newson & Krumm style)
    d_route = distance driven along the road network from the projection on arc a to the projection on arc b
              (same arc: along the arc; otherwise end of a -> shortest path -> start of b, bounded Dijkstra)
    no path found within the search limit -> fixed penalty UNREACHABLE (soft, so decoding never gets stuck)
Points without any candidate are skipped (no prediction). A time gap > gap_s breaks the chain:
the transition becomes uniform, so the route is re-started after a GPS dropout.
"""
import heapq
import numpy as np

UNREACHABLE = -30.0


class RoadGraph:
    """Directed node graph of one trip; edge weight = arc length (m)."""

    def __init__(self, u, v, arc_len, n_nodes):
        order = np.argsort(u, kind="stable")
        self.u, self.v, self.len = u, v, arc_len
        self.nbr_v, self.nbr_w = v[order], arc_len[order]
        self.ptr = np.searchsorted(u[order], np.arange(n_nodes + 1))
        self.cache = {}

    def dist_from(self, src, limit):
        """Shortest distances from node src to all nodes within `limit` metres (cached)."""
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for k in range(self.ptr[n], self.ptr[n + 1]):
                m, nd = self.nbr_v[k], d + self.nbr_w[k]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        self.cache[src] = (limit, dist)
        return dist


def prepare_trip(cands, px, py, t, graph, gap_s=60.0):
    """Precompute everything that does not depend on the decoder parameters.
    cands: DataFrame for one trip with columns point, arc, proj_frac, p."""
    cands = cands.sort_values(["point", "dist_rank"])
    steps = []  # (point, arcs, log_p_normalised, delta_matrix or None)
    prev = None
    for point, g in cands.groupby("point", sort=True):
        arcs = g.arc.values
        p = np.clip(g.p.values, 1e-6, None)
        logp = np.log(p / p.sum())
        frac = g.proj_frac.values
        delta = None
        if prev is not None and t[point] - t[prev["point"]] <= gap_s:
            d_gps = np.hypot(px[point] - px[prev["point"]], py[point] - py[prev["point"]])
            limit = 2 * d_gps + 200.0
            delta = np.full((len(prev["arcs"]), len(arcs)), np.nan)
            for ia, (a, fa) in enumerate(zip(prev["arcs"], prev["frac"])):
                la = graph.len[a]
                dist = None
                for ib, (b, fb) in enumerate(zip(arcs, frac)):
                    lb = graph.len[b]
                    if a == b:
                        d_route = abs(fb - fa) * la
                    elif graph.v[a] == graph.u[b]:
                        d_route = (1 - fa) * la + fb * lb
                    else:
                        if dist is None:
                            dist = graph.dist_from(graph.v[a], limit)
                        sp = dist.get(graph.u[b])
                        if sp is None:
                            continue  # unreachable -> stays NaN
                        d_route = (1 - fa) * la + sp + fb * lb
                    delta[ia, ib] = abs(d_route - d_gps)
        prev = {"point": point, "arcs": arcs, "frac": frac}
        steps.append((point, arcs, logp, delta))
    return steps


def viterbi(steps, beta=10.0, w_e=1.0):
    """Returns {point: predicted arc} for every point that had candidates."""
    if not steps:
        return {}
    scores = [w_e * steps[0][2]]
    backs = [None]
    for point, arcs, logp, delta in steps[1:]:
        prev_score = scores[-1]
        if delta is None:  # chain break (first point after a long gap): uniform transition
            best = np.full(len(arcs), int(np.argmax(prev_score)))
            sc = prev_score.max() + w_e * logp
        else:
            trans = np.where(np.isnan(delta), UNREACHABLE, -np.nan_to_num(delta) / beta)
            total = prev_score[:, None] + trans
            best = total.argmax(0)
            sc = total.max(0) + w_e * logp
        scores.append(sc - sc.max())  # keep numbers small
        backs.append(best)
    idx = int(np.argmax(scores[-1]))
    out = {}
    for k in range(len(steps) - 1, -1, -1):
        out[steps[k][0]] = steps[k][1][idx]
        if backs[k] is not None:
            idx = int(backs[k][idx])
    return out
