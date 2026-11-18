"""PathSnap evaluator UI (Laptop 2). Talks only to the API - it has no model and no ground truth.

Run:  py -3.13 -m streamlit run ui/app.py      (set the API address in the sidebar, e.g. http://<laptop-1-ip>:8000)
"""
import os
import requests
import folium
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map-matching and route reconstruction")
api = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")

try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach the API at {api}: {e}")
    st.stop()

auto = st.query_params.get("trip")  # optional link like ...?trip=00000000 runs that trip directly
trip = st.selectbox(f"Evaluation trip ({len(trips)} available)", trips,
                    index=trips.index(auto) if auto in trips else 0)
run_auto = auto in trips and st.session_state.get("auto_done") != auto
if st.button("Reconstruct route", type="primary") or run_auto:
    st.session_state["auto_done"] = auto
    with st.spinner(f"Sending trip {trip} to the API..."):
        r = requests.post(f"{api}/predict/{trip}", timeout=600)
    if r.ok:
        st.session_state["result"] = r.json()
    else:
        st.error(f"API error {r.status_code}: {r.text}")

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']} - model: {res['model']} + HMM/Viterbi")
    ev = res.get("evaluation", {})
    cols = st.columns(6)
    cols[0].metric("GPS points", res["gps_points"])
    cols[1].metric("Top-1 accuracy (HMM)", f"{ev['top1_accuracy_hmm']:.2%}" if ev else "n/a")
    cols[2].metric("Top-1 (classifier only)", f"{ev['top1_accuracy_classifier_only']:.2%}" if ev else "n/a")
    cols[3].metric("Route continuity", f"{ev['route_continuity']:.2%}" if ev else "n/a")
    cols[4].metric("Route F1", f"{ev['route_f1']:.2%}" if ev else "n/a")
    cols[5].metric("API inference time", f"{res['inference_seconds']:.1f} s")

    gps = res["gps"]
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    g_gps = folium.FeatureGroup(name="GPS observations")
    g_route = folium.FeatureGroup(name="Predicted route")
    step = max(1, len(gps) // 2000)
    for lat, lon in gps[::step]:
        folium.CircleMarker((lat, lon), radius=1.5, color="blue", fill=True, opacity=0.5).add_to(g_gps)
    for seg in res["predicted_route"]:
        folium.PolyLine(seg, color="red", weight=4, opacity=0.85).add_to(g_route)
    g_gps.add_to(m); g_route.add_to(m)  # route drawn on top of the GPS points
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:999;background:white;padding:6px;'
        'border:1px solid #999;font-size:13px"><span style="color:blue">&#9679;</span> GPS observations<br>'
        '<span style="color:red">&#9644;</span> Predicted route</div>'))
    components.html(m.get_root().render(), height=620)
    with st.expander("Raw API response (metrics only)"):
        st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_route")})
