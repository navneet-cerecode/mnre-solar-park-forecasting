"""India solar-park forecasting dashboard.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from solarpark import config, forecasts as F

st.set_page_config(page_title="MNRE Solar Park Forecasts", page_icon="☀️", layout="wide")
RES = config.REPORTS / "results"
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"
NOTE = "Test period Jan–Sep 2026. 'Actual' is pvlib-simulated from ERA5 weather (no park publishes hourly metered output)."


@st.cache_data
def fc(h):
    f = F.forecasts(h).copy()
    f["time_ist"] = F.to_ist(f.time_utc)
    return f


@st.cache_data
def agg(h, level, model):
    a = F.aggregate(h, level, model)
    a["time_ist"] = F.to_ist(a.time_utc)
    return a


@st.cache_data
def results(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def nrmse(y, p):
    return 100 * float(((p - y) ** 2).mean() ** 0.5)


def range_band(fig, x, lo, hi, name="80% range"):
    fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([hi, lo[::-1]]), fill="toself", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), name=name, hoverinfo="skip"))


def line_chart(df, x, cols, band=None, ytitle="MW"):
    fig = go.Figure()
    if band is not None and band[0] in df and df[band[0]].notna().any():
        g = df.dropna(subset=[band[0]])   # range is ~0 at dusk/dawn, so bridging the night is harmless
        range_band(fig, g[x], g[band[0]], g[band[1]])
    for col, name, color, dash in cols:
        g = df.set_index(x)[[col]].asfreq("h").reset_index()   # gaps at night instead of straight lines
        fig.add_trace(go.Scatter(x=g[x], y=g[col], name=name, line=dict(color=color, width=2 if dash == "solid" else 1.5, dash=dash)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=ytitle, legend=dict(orientation="h", y=1.08),
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------- pages
def overview():
    st.title("India's MNRE solar parks — short-term forecasts")
    st.caption(NOTE)
    parks = F.parks()
    h = st.sidebar.radio("Forecast horizon", [1, 3, 6], format_func=lambda v: f"{v} hour{'s' if v > 1 else ''} ahead")
    f = fc(h)
    e1 = results("e1_models")
    k = st.columns(5)
    k[0].metric("Sanctioned parks (official list)", "50", "37,990 MW", delta_color="off")
    k[1].metric("Parks forecast", f"{f.park_id.nunique()}", f"{int(parks[parks.tier == 'U2_core'].capacity_mw_model.sum()):,} MW commissioned", delta_color="off")
    if e1 is not None:
        r = e1[(e1.h == h) & (e1.subset == "all 43 parks")].set_index("model")
        k[2].metric("Park error (LightGBM)", f"{r.loc['lightgbm', 'nRMSE_%']:.2f}%", f"{100 * r.loc['lightgbm', 'skill_vs_smart']:.0f}% better than baseline")
    a = agg(h, None, "lightgbm")
    tot = a[a.actual_mw > 0]
    cap = f.drop_duplicates("park_id").capacity_mw.sum()
    k[3].metric("All-parks total error", f"{nrmse(tot.actual_mw, tot.forecast_mw) / cap:.2f}%", "of total capacity", delta_color="off")
    hw = F.total_halfwidth_mw(h)
    k[4].metric("80% range, all parks", f"± {hw:,.0f} MW" if hw else "–", "calibrated (conformal)", delta_color="off")

    days = sorted(f.time_ist.dt.date.unique())
    c1, c2 = st.columns([1, 1])
    day = c1.select_slider("Day", options=days, value=days[min(len(days) - 1, 150)])
    hours = sorted(f[f.time_ist.dt.date == day].time_ist.dt.hour.unique())
    hour = c2.select_slider("Hour (IST, end of hour)", options=hours, value=min(hours, key=lambda x: abs(x - 13)))

    snap = f[(f.time_ist.dt.date == day) & (f.time_ist.dt.hour == hour)][["park_id", "y_mw", "lightgbm_mw", "lo_mw", "hi_mw"]]
    m = parks.merge(snap, on="park_id", how="left").dropna(subset=["latitude"])
    err = f.groupby("park_id").apply(lambda g: nrmse(g.y[g.elevation > 5], g.lightgbm[g.elevation > 5]), include_groups=False).rename("park_nRMSE")
    m = m.merge(err, on="park_id", how="left")
    m["label"] = m.apply(lambda r: f"{r.park_name} ({r.state})", axis=1)
    m["Actual MW"] = m.y_mw.round(0)
    m["Forecast MW"] = m.lightgbm_mw.round(0)
    m["80% range"] = m.apply(lambda r: f"{r.lo_mw:,.0f}–{r.hi_mw:,.0f} MW" if pd.notna(r.lo_mw) else "–", axis=1)
    m["Test error %"] = m.park_nRMSE.round(2)
    m["size"] = m.sanctioned_capacity_mw.clip(lower=50)
    color = st.radio("Colour markers by", ["status", "test error %"], horizontal=True)
    fig = px.scatter_map(m, lat="latitude", lon="longitude", size="size", size_max=28, zoom=3.9, height=600,
                         color_discrete_map={"Completed": BLUE, "Partially commissioned": ORANGE, "Under implementation": GREEN,
                                             "Dropped/revised by 2026": GREY},
                         center=dict(lat=22.5, lon=80), map_style="carto-positron", hover_name="label",
                         color="status" if color == "status" else "Test error %",
                         color_continuous_scale="Blues" if color != "status" else None,
                         hover_data={"sanctioned_capacity_mw": True, "Actual MW": True, "Forecast MW": True, "80% range": True,
                                     "Test error %": True, "size": False, "latitude": False, "longitude": False})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.02, x=0.01))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})   # page scroll must not zoom the map
    st.caption("Boundaries and park coordinates are approximate (OpenStreetMap geocoding). Parks without a verified location are not shown.")

    st.subheader(f"All parks combined — {day}")
    d = a[F.to_ist(a.time_utc).dt.date == day]
    st.plotly_chart(line_chart(d, "time_ist", [("actual_mw", "Actual (simulated)", "#0b0b0b", "solid"),
                                               ("forecast_mw", f"LightGBM, {h} h ahead", BLUE, "solid")], band=("lo_mw", "hi_mw")), use_container_width=True)


def park_explorer():
    st.title("Solar park explorer")
    st.caption(NOTE)
    parks = F.parks()
    f = fc(1)
    modelled = parks[parks.park_id.isin(f.park_id.unique())]
    c = st.columns(3)
    state = c[0].selectbox("State", sorted(modelled.state.unique()))
    sel = modelled[modelled.state == state]
    pid = c[1].selectbox("Park", sel.park_id, format_func=lambda p: sel.set_index("park_id").park_name[p])
    h = c[2].radio("Horizon", [1, 3, 6], horizontal=True, format_func=lambda v: f"{v} h")
    p = parks.set_index("park_id").loc[pid]
    info = st.columns(4)
    info[0].metric("Sanctioned", f"{p.sanctioned_capacity_mw:,} MW")
    info[1].metric("Modelled capacity", f"{p.capacity_mw_model:,.0f} MW", "commissioned" if p.tier == "U2_core" else "virtual (sanctioned)", delta_color="off")
    info[2].metric("Status", p.status)
    info[3].metric("Developer (SPPD)", str(p.sppd).split(" (")[0][:28])

    g = fc(h)
    g = g[g.park_id == pid].sort_values("time_ist")
    lo_d, hi_d = g.time_ist.dt.date.min(), g.time_ist.dt.date.max()
    start = st.date_input("Week starting", value=pd.Timestamp("2026-07-06").date(), min_value=lo_d, max_value=hi_d)
    w = g[(g.time_ist.dt.date >= start) & (g.time_ist.dt.date < start + pd.Timedelta(days=7))]
    show = st.multiselect("Forecasts", list(F.MODELS), default=["lightgbm", "gru", "bl_smart"], format_func=F.MODELS.get)
    palette = {"lightgbm": BLUE, "gru": ORANGE, "ensemble_lgbm_gru": GREEN, "bl_smart": GREY}
    cols = [("y_mw", "Actual (simulated)", "#0b0b0b", "solid")] + [(f"{m}_mw", F.MODELS[m], palette[m], "solid" if m == "lightgbm" else "dash") for m in show]
    st.plotly_chart(line_chart(w, "time_ist", cols, band=("lo_mw", "hi_mw")), use_container_width=True)
    day = g[g.elevation > 5]
    st.subheader("Accuracy over the whole test period (daylight hours, % of capacity)")
    st.dataframe(pd.DataFrame([{"Forecast": F.MODELS[m], "nRMSE %": round(nrmse(day.y, day[m]), 2),
                                "nMAE %": round(100 * (day[m] - day.y).abs().mean(), 2)} for m in F.MODELS]), hide_index=True)


def states_india():
    st.title("State, region and all-India totals")
    st.caption(NOTE)
    c = st.columns(3)
    h = c[0].radio("Horizon", [1, 3, 6], horizontal=True, format_func=lambda v: f"{v} h")
    level = c[1].radio("Level", ["All parks", "Region", "State"], horizontal=True)
    lv = {"All parks": None, "Region": "region", "State": "state"}[level]
    a = agg(h, lv, "lightgbm")
    if lv:
        name = c[2].selectbox(level, sorted(a[lv].unique()))
        a = a[a[lv] == name]
    start = st.date_input("Week starting", value=pd.Timestamp("2026-07-06").date())
    w = a[(a.time_ist.dt.date >= start) & (a.time_ist.dt.date < start + pd.Timedelta(days=7))]
    st.plotly_chart(line_chart(w, "time_ist", [("actual_mw", "Actual (simulated)", "#0b0b0b", "solid"),
                                               ("forecast_mw", f"LightGBM, {h} h ahead", BLUE, "solid")], band=("lo_mw", "hi_mw")),
                    use_container_width=True)
    if lv is None:
        st.caption("Shaded: calibrated 80% range for the all-parks total (split conformal, experiment E8).")
    else:
        st.caption("No calibrated range is shown for states/regions (ranges cannot simply be added across parks).")
    e5 = results("e5_aggregation")
    if e5 is not None:
        st.subheader("Adding parks together reduces the error (experiment E5)")
        st.dataframe(e5.round(2), hide_index=True)


def model_results():
    st.title("Model results and experiments")
    st.caption(NOTE)
    for col, (fig, cap) in zip(st.columns([3, 2]), [("fig_e1_models.png", "E1: error by model and horizon"),
                                                    ("fig_e5_aggregation.png", "E5: error falls as parks are added together")]):
        if (RES / fig).exists():
            col.image(str(RES / fig), caption=cap)
    names = {"e1_models": "E1 – models (trees, baselines)", "e1b_deep_learning": "E1 – deep learning (GPU)",
             "e2_per_park_vs_global": "E2 – global vs per-park", "e3_regional_vs_global": "E3 – regional vs global",
             "e4_ablation": "E4 – which inputs matter", "e4_shap_share": "E4 – SHAP share", "e5_aggregation": "E5 – aggregation",
             "e5_error_corr_vs_distance": "E5 – error correlation vs distance", "e6_ramps_regimes": "E6 – ramps & weather",
             "e7_unseen_locations": "E7 – unseen locations", "e8_intervals": "E8 – prediction intervals",
             "e8_cqr_monthly_coverage": "E8 – coverage by month", "e9_oracle_vs_nwp": "E9 – forecast vs perfect weather",
             "e10_catboost_tuning": "E10 – CatBoost (GPU) and LightGBM tuning"}
    pick = st.selectbox("Table", [n for n in names if (RES / f"{n}.csv").exists()], format_func=names.get)
    st.dataframe(results(pick).round(3), hide_index=True, use_container_width=True)


st.navigation([st.Page(overview, title="India overview", icon="🗺️"), st.Page(park_explorer, title="Park explorer", icon="☀️"),
               st.Page(states_india, title="State & India totals", icon="📈"), st.Page(model_results, title="Model results", icon="🧪")]).run()
