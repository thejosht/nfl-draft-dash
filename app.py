# --- imports & theme ---
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Keep your fixed palette (works on older Plotly too by applying per-figure)
COLORWAY = ['#7cc0ff', '#f7b267', '#59d499', '#e76f51', '#957fef', '#ff6b6b', '#2a9d8f']
# (Optional) also set Plotly defaults so you don't have to pass color sequences each time
px.defaults.color_discrete_sequence = COLORWAY

# --- load data (all CSVs expected in this folder) ---
@st.cache_data(show_spinner=False)
def must_read(name: str) -> pd.DataFrame:
    """Read a CSV from the repo folder; warn if missing; tolerant engine/encoding for Cloud."""
    p = Path(name)
    if not p.exists():
        st.error(f"Missing file: {name}")
        return pd.DataFrame()
    try:
        # fast path if pyarrow is available
        return pd.read_csv(p, engine="pyarrow")
    except Exception:
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception:
            return pd.read_csv(p, low_memory=False, encoding="latin-1")

kpi   = must_read("kpi_snapshot.csv")
ev    = must_read("ev_line_ribbon.csv")          # pick, ev, rib_low, rib_high, pos_group, era
heat  = must_read("surplus_heatmap.csv")         # era, pick_band, pos_group, mean_surplus
price = must_read("price_vs_performance.csv")    # apy_m, av_y1_4, pos_group, player_name, draft_year, team, pick, era
roll  = must_read("learning_over_time.csv")      # draft_year, pick_band, surplus_3yr, era
teams = must_read("teams_roi_surplus.csv")       # team, era, pick_band, mean_surplus, median_roi (or roi)

# Optional files (used only if present)
forecast_path = Path("learning_forecast.csv")
clusters_path = Path("player_clusters.csv")

# --- KPI helper ---
def kpi_from(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame([{
            "players": 0,
            "Avg APY ($M)": 0.0,
            "Avg AV (Y1–4)": 0.0,
            "Median ROI": 0.0,
            "Mean Surplus": 0.0
        }])
    out = pd.DataFrame([{
        "players":      int(df["player_name"].count()) if "player_name" in df else int(len(df)),
        "Avg APY ($M)": float(df["apy_m"].mean(skipna=True)) if "apy_m" in df else 0.0,
        "Avg AV (Y1–4)":float(df["av_y1_4"].mean(skipna=True)) if "av_y1_4" in df else 0.0,
        "Median ROI":   float(df["roi"].median(skipna=True)) if "roi" in df else 0.0,
        "Mean Surplus": float(df["mean_surplus"].mean(skipna=True)) if "mean_surplus" in df else 0.0,
    }])
    return out

# --- page setup ---
st.set_page_config(page_title="NFL Draft — Price vs Performance", layout="wide")
st.title("Is the NFL Draft an Efficient Market? — Price vs Performance (Rounds 1–3, 2006–2024)")

# Guard early if any core table is missing
core_missing = any(d.empty for d in [ev, heat, price, roll, teams])
if core_missing:
    st.stop()

# ========================== FILTERS (era/pos/team/round/bands) ==========================
BAND_ORDER = ["01–08", "09–16", "17–32", "33–64", "65–96+"]

# ---------- helpers ----------
def _ensure_round_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add nullable round from pick: 1=1–32, 2=33–64, 3=65–96; NA otherwise."""
    if not isinstance(df, pd.DataFrame) or df.empty or "round" in df.columns:
        return df
    if "pick" not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out["pick"], errors="coerce")
    r = np.where(p <= 32, 1,
         np.where(p <= 64, 2,
         np.where(p <= 96, 3, np.nan)))
    out["round"] = pd.Series(r, index=out.index).astype("Int64")
    return out

def _ensure_pick_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add pick_band (01–08 … 65–96+) from pick; NA-safe; last band is open-ended."""
    if not isinstance(df, pd.DataFrame) or df.empty or "pick_band" in df.columns:
        return df
    if "pick" not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out["pick"], errors="coerce")
    bins   = [0, 8, 16, 32, 64, np.inf]  # 5 intervals for 5 labels
    labels = BAND_ORDER
    out["pick_band"] = pd.cut(
        p, bins=bins, labels=labels, include_lowest=True, right=True
    ).astype("object")
    return out

def _unique_from(dfs, col):
    vals = []
    for d in dfs:
        if isinstance(d, pd.DataFrame) and not d.empty and col in d.columns:
            vals.append(d[col])
    if not vals:
        return []
    return sorted(pd.concat(vals, ignore_index=True).dropna().unique().tolist())

def _round_opts_from(*dfs) -> list[int]:
    have = set()
    for df in dfs:
        if isinstance(df, pd.DataFrame) and not df.empty:
            d = _ensure_round_col(df)
            if "round" in d.columns:
                have.update(pd.Series(d["round"]).dropna().astype(int).unique().tolist())
    opts = sorted([r for r in have if r in (1, 2, 3)])
    return opts if opts else [1, 2, 3]

def _band_opts_from(heat_df, roll_df, price_df, teams_df) -> list[str]:
    found = set()
    if isinstance(heat_df, pd.DataFrame) and not heat_df.empty:
        # also allow alt column names, then rename in the filter
        cols = [c if isinstance(c, str) else str(c) for c in heat_df.columns]
        for c in cols:
            if c in BAND_ORDER:
                found.add(c)
    for df in (roll_df, price_df, teams_df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            df2 = _ensure_pick_band(df)
            if "pick_band" in df2.columns:
                vals = pd.Series(df2["pick_band"]).dropna().astype(str).unique().tolist()
                found.update(v for v in vals if v in BAND_ORDER)
    opts = [b for b in BAND_ORDER if b in found]
    return opts if opts else BAND_ORDER.copy()

def _apply_filters_long(df, era_sel, pos_sel, team_sel, rnd_sel):
    """Filter a long table on era/pos/team/round (if columns exist)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    if "era" in out.columns and era_sel:
        out = out[out["era"].isin(era_sel)]

    if "pos_group" in out.columns and pos_sel:
        out = out[out["pos_group"].isin(pos_sel)]

    if "team" in out.columns and team_sel:
        out = out[out["team"].isin(team_sel)]

    out = _ensure_round_col(out)
    if "round" in out.columns and rnd_sel:
        out = out[out["round"].astype("Int64").isin(pd.Series(rnd_sel, dtype="Int64"))]

    # Keep pick_band for downstream usage
    out = _ensure_pick_band(out)
    return out

def _filter_heat_wide(heat_df, era_sel, pos_sel, band_sel):
    """
    Filter heatmap wide table robustly:
      - Rename 'Position Group'->'pos_group' and 'Era'->'era'
      - Filter rows on era/pos if present
      - Keep at least one band column (don’t trim to zero)
    """
    if not isinstance(heat_df, pd.DataFrame) or heat_df.empty:
        return heat_df
    out = heat_df.copy()

    # Normalize id column names
    out = out.rename(columns={"Position Group": "pos_group", "Era": "era"})

    # Row filters
    if "era" in out.columns and era_sel:
        out = out[out["era"].isin(era_sel)]
    if "pos_group" in out.columns and pos_sel:
        out = out[out["pos_group"].isin(pos_sel)]

    # Available band columns that still exist
    available_bands = [b for b in BAND_ORDER if b in out.columns]

    # If user selected bands, intersect; otherwise keep all available
    if band_sel:
        bands_to_keep = [b for b in available_bands if b in band_sel]
        # If intersection is empty (e.g., due to mismatched labels), keep available
        if bands_to_keep:
            keep_cols = bands_to_keep
        else:
            keep_cols = available_bands
    else:
        keep_cols = available_bands

    # If there are no band columns at all, just return with id cols — your downstream
    # chart should detect and show a friendly message.
    id_cols = [c for c in out.columns if c in ("pos_group", "era")]
    if keep_cols:
        out = out[id_cols + keep_cols]
    else:
        out = out[id_cols]

    return out

# ---------- build UI options ----------
pos_opts   = _unique_from([price, ev, roll, teams], "pos_group")
era_opts   = _unique_from([teams, price, ev, roll], "era")
team_opts  = _unique_from([teams, price, ev, roll], "team")   # NEW: team list
band_opts  = _band_opts_from(heat, roll, price, teams)
round_opts = _round_opts_from(price, ev, roll, teams)

with st.sidebar:
    st.header("Filters")
    era_sel   = st.multiselect("Era", era_opts, default=era_opts)
    pos_sel   = st.multiselect("Position Group", pos_opts, default=pos_opts)
    team_sel  = st.multiselect("Team", team_opts, default=team_opts)  # NEW: team filter
    rnd_sel   = st.multiselect("Rounds (1–3)", round_opts, default=round_opts,
                               help="Derived from pick #; 1=1–32, 2=33–64, 3=65–96.")
    band_sel  = st.multiselect("Pick bands (for Heatmap & Trend)",
                               band_opts, default=band_opts)

# ---------- apply to each table ----------
price_f = _apply_filters_long(price, era_sel, pos_sel, team_sel, rnd_sel)
ev_f    = _apply_filters_long(ev,    era_sel, pos_sel, team_sel, rnd_sel)
roll_f  = _apply_filters_long(roll,  era_sel, pos_sel, team_sel, rnd_sel)
teams_f = _apply_filters_long(teams, era_sel, pos_sel, team_sel, rnd_sel)
heat_f  = _filter_heat_wide(heat,    era_sel, pos_sel, band_sel)
# ======================== END FILTERS ========================

# --- KPI STRIP (full replacement) ------------------------------------------------
# Chooses the best table for APY/AV/ROI, computes KPIs, and renders friendly labels.

def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

# 1) Pick a metrics table that actually has APY/AV/ROI
metrics_f = price_f if (isinstance(price_f, pd.DataFrame) and not price_f.empty) \
            else (ev_f if (isinstance(ev_f, pd.DataFrame) and not ev_f.empty) \
            else (roll_f if (isinstance(roll_f, pd.DataFrame) and not roll_f.empty) else pd.DataFrame()))

# 2) Build the base KPI row
if metrics_f.empty:
    kpi = pd.DataFrame([{
        "players": 0,
        "Avg APY ($M)": 0.0,
        "Avg AV (Y1–4)": 0.0,
        "Median ROI": 0.0,
        "Mean Surplus": 0.0,
    }])
else:
    kpi = kpi_from(metrics_f).round(2)
    # Be defensive: ensure it's a single-row frame
    if isinstance(kpi, pd.Series):
        kpi = kpi.to_frame().T

# 3) Overwrite KPI “Mean Surplus” from aggregated teams table (unweighted)
if isinstance(teams_f, pd.DataFrame) and not teams_f.empty:
    # Try common column names for surplus in teams table
    for cand in ("surplus", "mean_surplus"):
        if cand in teams_f.columns:
            ms = pd.to_numeric(teams_f[cand], errors="coerce").mean()
            if pd.notna(ms):
                # Ensure the KPI frame has a target column to write to
                target = None
                for c in kpi.columns:
                    if _norm(c) in {"meansurplus", "surplusmean"}:
                        target = c
                        break
                if target is None:
                    target = "Mean Surplus"
                    if target not in kpi.columns:
                        kpi[target] = 0.0
                kpi.loc[:, target] = round(float(ms), 2)
            break

# 4) Resolve column names → canonical keys → friendly titles
#    We allow multiple possible raw names so we work with any kpi_from() flavor.
raw_cols = { _norm(c): c for c in kpi.columns }

def _find(*candidates):
    # return the actual kpi column matching any of these names
    wanted = {_norm(x) for x in candidates}
    for norm_name, real_name in raw_cols.items():
        if norm_name in wanted:
            return real_name
    return None

players_col  = _find("players")
apy_col      = _find("avgapy","avgapy$m","avgapy$m)", "avgapy($m)", "avgapym") or _find("avgapy($m)","avgapy(dollars)","avgapy")
av_col       = _find("avgav(y14)","avgy14","avy14","avgav(y1–4)","avgav(y1-4)")
roi_col      = _find("medianroi","roi")
surplus_col  = _find("meansurplus","surplusmean")

# Pull values (default to 0 if any column is missing)
row = kpi.iloc[0] if len(kpi) else pd.Series(dtype="float")
def _get(col): 
    return float(pd.to_numeric(row.get(col, 0), errors="coerce")) if col else 0.0

players_val  = int(_get(players_col))
apy_val      = _get(apy_col)
av_val       = _get(av_col)
roi_val      = _get(roi_col)
surplus_val  = _get(surplus_col)

# 5) Friendly titles & display formatting (matches your mockup)
friendly = [
    ("Players in view",            players_val, lambda v: f"{int(v):,}"),
    ("Avg APY ($/yr)",             apy_val,     lambda v: f"${v:.2f}M"),
    ("Avg AV (Y1–4)",              av_val,      lambda v: f"{v:.2f}"),
    ("ROI (AV per $1M)",           roi_val,     lambda v: f"{v:.2f}"),
    ("Avg Surplus (AV − EV)",      surplus_val, lambda v: f"{v:+.2f}"),
]

# 6) Render 5 metrics
c1, c2, c3, c4, c5 = st.columns(5)
for col, (label, value, fmt) in zip([c1, c2, c3, c4, c5], friendly):
    try:
        col.metric(label, fmt(value))
    except Exception:
        col.metric(label, str(value))

st.markdown("---")
# --- end KPI STRIP ------------------------------------------------------------

# ===================== CALL OUTS & GUIDE (expander) =====================
with st.expander("CALLOUTS & GUIDE", expanded=False):

    # ---------- Small CSS: badges + responsive math ----------
    st.markdown(
        """
        <style>
          .band-badge{
            display:inline-block; padding:2px 10px; margin:2px 6px 2px 0;
            border-radius:18px; font-weight:600; font-size:0.95rem;
            border:1px solid rgba(255,255,255,0.15);
          }
          .good{ background:#0f5132; color:#d1f7e6; border-color:#198754; }
          .bad{  background:#5c2020; color:#ffe0e0; border-color:#dc3545; }
          .warn{ background:#4a3a13; color:#ffe6b3; border-color:#d39e00; }
          .dim{  opacity:0.95; }

          /* Make KaTeX (st.latex) formulas adapt to smaller widths */
          .math-wrap .katex { font-size: 1.25rem; }
          @media (max-width: 1400px){ .math-wrap .katex{ font-size:1.08rem; } }
          @media (max-width: 1200px){ .math-wrap .katex{ font-size:0.98rem; } }
          @media (max-width: 1000px){ .math-wrap .katex{ font-size:0.90rem; } }
          @media (max-width: 820px) { .math-wrap .katex{ font-size:0.82rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Helpers for callouts ----------
    BAND_ORDER = ["01–08", "09–16", "17–32", "33–64", "65–96+"]

    def sort_bands_like_app(index_like):
        order = {b: i for i, b in enumerate(BAND_ORDER)}
        return sorted([b for b in index_like if pd.notna(b)], key=lambda x: order.get(x, 999))

    def add_pick_band(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a pick_band column exists (derived from 'pick' if needed)."""
        if df is None or df.empty:
            return df
        if "pick_band" in df.columns:
            return df
        if "pick" not in df.columns:
            return df
        out = df.copy()
        p = pd.to_numeric(out.get("pick"), errors="coerce")
        bins = [0, 8, 16, 32, 64, 10_000]
        out["pick_band"] = pd.cut(p, bins=bins, labels=BAND_ORDER, right=True)
        return out

    def badge(txt, kind="good"):
        kind = kind if kind in {"good", "bad", "warn"} else "warn"
        return f'<span class="band-badge {kind}">{txt}</span>'

    def top_bottom(series, k=2, larger_is_better=True):
        """Return top k and bottom k bands from a Series indexed by band."""
        if series is None or series.empty:
            return [], []
        s = series.copy().dropna()
        if s.empty:
            return [], []
        s = s.loc[[b for b in sort_bands_like_app(s.index) if b in s.index]]
        s_sorted = s.sort_values(ascending=not larger_is_better)
        tops = list(s_sorted.head(k).index)
        bots = list(s.sort_values(ascending=larger_is_better).head(k).index)
        return tops, bots

    # ---------- Aggregate metrics (respect current filters) ----------
    # Surplus by band (prefer heat_f from heatmap; then teams_f; else compute from price_f)
    surplus_by_band = None
    if isinstance(heat_f, pd.DataFrame) and not heat_f.empty and {"pick_band","mean_surplus"}.issubset(heat_f.columns):
        surplus_by_band = heat_f.groupby("pick_band")["mean_surplus"].mean()
    elif isinstance(teams_f, pd.DataFrame) and not teams_f.empty and {"pick_band","surplus"}.issubset(teams_f.columns):
        surplus_by_band = teams_f.groupby("pick_band")["surplus"].median()
    elif isinstance(price_f, pd.DataFrame) and not price_f.empty and "surplus" in price_f.columns:
        pf_tmp = add_pick_band(price_f)
        if "pick_band" in pf_tmp.columns:
            surplus_by_band = pf_tmp.groupby("pick_band")["surplus"].median()

    # ROI by band (from player-level price_f)
    roi_by_band = None
    if isinstance(price_f, pd.DataFrame) and not price_f.empty and {"av_y1_4","apy_m"}.issubset(price_f.columns):
        pf = add_pick_band(price_f)
        if "pick_band" in pf.columns:
            t = pf.replace([np.inf, -np.inf], np.nan).dropna(subset=["apy_m","av_y1_4","pick_band"]).copy()
            t = t[t["apy_m"] > 0]
            if not t.empty:
                t["roi_calc"] = t["av_y1_4"] / t["apy_m"]
                roi_by_band = t.groupby("pick_band")["roi_calc"].median()

    # ---------- UI: how many bands to show per side ----------
    k_callouts = st.slider(
        "How many bands per side for callouts",
        min_value=1, max_value=3, value=2, step=1, key="k_callouts_v2"
    )

    # Compute callouts
    surplus_tops, surplus_bottoms = top_bottom(surplus_by_band, k=k_callouts, larger_is_better=True)
    roi_tops, roi_bottoms         = top_bottom(roi_by_band,     k=k_callouts, larger_is_better=True)

    # ---------- Render callouts (two columns) ----------
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Sweet spots")
        good_s = " ".join(badge(b, "good") for b in surplus_tops) if surplus_tops else badge("n/a","warn")
        good_r = " ".join(badge(b, "good") for b in roi_tops)     if roi_tops     else badge("n/a","warn")
        st.markdown(f'by **Surplus**: {good_s} • by **ROI**: {good_r}', unsafe_allow_html=True)

    with c_right:
        st.subheader("Money pits")
        bad_s = " ".join(badge(b, "bad") for b in surplus_bottoms) if surplus_bottoms else badge("n/a","warn")
        bad_r = " ".join(badge(b, "bad") for b in roi_bottoms)     if roi_bottoms     else badge("n/a","warn")
        st.markdown(f'by **Surplus**: {bad_s} • by **ROI**: {bad_r}', unsafe_allow_html=True)

    st.divider()

    # ---------- Key terms + Formulas (responsive) ----------
    key_col, formula_col = st.columns([1,1])
    with key_col:
        st.subheader("Key terms")
        st.markdown(
            """
- **AV (Approximate Value)** – on-field contribution (we use Years **1–4**).
- **Price (APY)** – Average Pay per Year on the rookie deal ($/yr).
- **EV (Expected Value)** – typical AV for a pick within its position; uses a **smoothed pick→AV curve** that is **era-aware**.
- **ROI** – AV delivered per **$1M** of APY.
- **Surplus** – value vs expectation at that pick and position (**positive = better**).
- **Pick bands** – 01–08, 09–16, 17–32, 33–64, 65–96+.
            """
        )

    with formula_col:
        st.subheader("Formulas")
        st.markdown('<div class="math-wrap">', unsafe_allow_html=True)
        st.latex(r"ROI \;=\; \frac{AV_{Y1\text{–}4}}{APY_{\$M}}")
        st.latex(r"Surplus \;=\; AV_{Y1\text{–}4} \;-\; EV_{\text{era}}(\text{pick},\;\text{pos\_group})")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ---------- About this dashboard ----------
    st.subheader("About this dashboard")
    st.markdown(
        """
**Goal:** Show where the NFL draft market **over or under prices** players by comparing **Price (APY)** with **Performance (AV, Years 1–4)** against an **era and position aware Expected Value (EV)** curve.

**How to use:** Pick an **Era**, **Team**, and **Rounds (1–3)** on the left. Then read: **EV by Pick** (confidence ribbon), **Surplus heatmap** (pick band × position), **Price ↔ Performance** scatter, **Learning over time**, and **Top/Bottom** teams. The **Sweet spots** and **Money pits** above summarize the best/worst pick bands by **Surplus** and **ROI**.

**Who is this for:** Anyone who wants a quick, evidence based view of **draft value**.

*Hover charts for details; controls are labeled on their tiles.*
        """
    )
# ================== end CALL OUTS & GUIDE (expander) ==================

# ============ 1) EV by Pick (line + ribbon) ============
st.subheader("EV by Pick (line + ribbon)")
st.caption("This shows the expected AV by draft pick for the selected positions. The line is the typical value; the ribbon is the usual range. Use it to see how value drops across the first 96 picks and how positions differ.")
if ev_f.empty:
    st.info("No EV data for current filters.")
else:
    fig = go.Figure()
    for pg, g in ev_f.groupby(["pos_group","era"]):
        g = g.sort_values("pick")
        # main EV line
        fig.add_trace(go.Scatter(
            x=g["pick"], y=g["ev"], mode="lines",
            name=f"EV • {pg}", hoverinfo="skip"
        ))
        # ribbon (uncertainty)
        fig.add_trace(go.Scatter(
            x=pd.concat([g["pick"], g["pick"][::-1]]),
            y=pd.concat([g["rib_high"], g["rib_low"][::-1]]),
            fill="toself", fillcolor="rgba(120,100,200,0.15)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name=f"EV band • {pg}", showlegend=False
        ))
    fig.update_layout(
        xaxis_title="Pick", yaxis_title="Expected AV (Y1–4)",
        height=380, margin=dict(l=20,r=20,t=30,b=20)
    )
    fig.update_layout(colorway=COLORWAY)
    st.plotly_chart(fig, width="stretch")


# ============ 2) Surplus heatmap (robust) ==========================
st.subheader("Surplus heatmap — mean Surplus by Pick band × Position Group")
st.caption("Average Surplus = AV − EV by pick band and position. Green = better than expected; red = worse. Use it to spot “sweet spots” where teams have historically found value.")

import numpy as np

# Canonical pick–band labels (match your other tiles)
BAND_ORDER = ["01–08", "09–16", "17–32", "33–64", "65–96+"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Bring id column names to canonical ('pos_group','era')."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    return df.rename(columns={"Position Group": "pos_group", "Era": "era"})

def _ensure_pick_band_local(df: pd.DataFrame) -> pd.DataFrame:
    """Add pick_band from 'pick' if needed, using canonical labels."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "pick_band" in df.columns:
        return df
    if "pick" not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out["pick"], errors="coerce")
    out["pick_band"] = pd.cut(
        p, bins=[0, 8, 16, 32, 64, 10_000], labels=BAND_ORDER, right=True
    )
    return out

def _maybe_build_wide_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    From a LONG table containing pos_group, pick_band and surplus/mean_surplus,
    build a WIDE table (id cols + band columns). Era is preserved if present.
    """
    if not isinstance(df_long, pd.DataFrame) or df_long.empty:
        return pd.DataFrame()
    if not {"pos_group", "pick_band"}.issubset(df_long.columns):
        return pd.DataFrame()

    val = (
        "mean_surplus" if "mean_surplus" in df_long.columns
        else ("surplus" if "surplus" in df_long.columns else None)
    )
    if val is None:
        return pd.DataFrame()

    id_cols = ["pos_group"] + (["era"] if "era" in df_long.columns else [])
    g = (
        df_long.groupby(id_cols + ["pick_band"], dropna=True)[val]
        .mean()
        .reset_index()
    )
    wide = g.pivot(index=id_cols, columns="pick_band", values=val).reset_index()

    # Ensure canonical band columns exist & order
    for b in BAND_ORDER:
        if b not in wide.columns:
            wide[b] = np.nan
    wide = wide[id_cols + BAND_ORDER]
    return wide

def _ensure_heat_wide() -> pd.DataFrame:
    """
    1) Try the provided heat_f (wide or long).
    2) Else rebuild from price_f (long) or teams_f (long).
    Returns a wide table with at least one band column if possible.
    """
    # 1) Try heat_f
    if "heat_f" in locals() and isinstance(heat_f, pd.DataFrame) and not heat_f.empty:
        h = _normalize_cols(heat_f.copy())
        if "pos_group" in h.columns and any(b in h.columns for b in BAND_ORDER):
            wide = h.copy()
        else:
            wide = _maybe_build_wide_from_long(h)
        if isinstance(wide, pd.DataFrame) and not wide.empty:
            return wide

    # 2) Try long from price_f then teams_f
    for candidate in [globals().get("price_f"), globals().get("teams_f")]:
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            long_src = _ensure_pick_band_local(candidate)
            wide = _maybe_build_wide_from_long(long_src)
            if isinstance(wide, pd.DataFrame) and not wide.empty:
                return wide

    return pd.DataFrame()

# Build a robust wide heat table
wide = _ensure_heat_wide()

# Apply sidebar filters if they exist
if isinstance(wide, pd.DataFrame) and not wide.empty:
    if "era_sel" in locals() and "era" in wide.columns and era_sel:
        wide = wide[wide["era"].isin(era_sel)]
    if "pos_sel" in locals() and pos_sel:
        wide = wide[wide["pos_group"].isin(pos_sel)]

    # Respect band selection, but never trim to zero bands
    band_cols_available = [b for b in BAND_ORDER if b in wide.columns]
    if "band_sel" in locals() and band_sel:
        band_cols_keep = [b for b in band_cols_available if b in band_sel]
        if band_cols_keep:
            band_cols_available = band_cols_keep

    # Prepare LONG display table then pivot to pos_group × pick_band
    if not band_cols_available:
        st.info("No surplus values available for current selection.")
    else:
        long_disp = (
            wide.melt(
                id_vars=[c for c in ["pos_group", "era"] if c in wide.columns],
                value_vars=band_cols_available,
                var_name="pick_band",
                value_name="surplus",
            )
        )
        # If multiple eras are present, collapse to mean for display
        if "era" in long_disp.columns and long_disp["era"].nunique() > 1:
            long_disp = (
                long_disp.groupby(["pos_group", "pick_band"])["surplus"]
                .mean()
                .reset_index()
            )

        pivot = (
            long_disp.pivot(index="pos_group", columns="pick_band", values="surplus")
            .reindex(columns=band_cols_available)
            .sort_index()
        )

        if pivot.empty or pivot.shape[1] == 0:
            st.info("No surplus values available for current selection.")
        else:
            import plotly.express as px
            fig = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                origin="upper",
                aspect="auto",
                labels=dict(color="Mean Surplus"),
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")
else:
    st.info("No surplus values available for current selection.")
# ===============================================================

# ============= 3) Price ↔ Performance (APY vs AV) ============
from pathlib import Path

st.subheader("Price ↔ Performance — APY ($M) vs AV (Y1–4)")
st.caption("Each dot is a player’s rookie price (APY, $M) vs performance (AV Y1–4). Hover for name/team/year/pick. Look for high-AV at low APY (great value) and low-AV at high APY (overpay). Toggle Clusters to group similar outcomes if available.")


pp = price_f.dropna(subset=["apy_m", "av_y1_4"]) if "price_f" in globals() else pd.DataFrame()
if pp.empty:
    st.info("No scatter points for current filters (likely missing APY).")
else:
    # Always show the toggle; if the file is missing we fall back gracefully.
    use_clusters = st.toggle("Color by value archetype (clusters)", value=False)
    clusters_path = Path("player_clusters.csv")
    has_clusters = clusters_path.exists()

    if use_clusters:
        if has_clusters:
            cl = pd.read_csv(clusters_path)
            needed = ["player_name", "draft_year", "cluster"]
            if all(c in cl.columns for c in needed):
                pp = pp.merge(cl[needed], on=["player_name", "draft_year"], how="left")
                pp["cluster"] = pd.to_numeric(pp["cluster"], errors="coerce").astype("Int64").astype(str)
                fig = px.scatter(
                    pp, x="apy_m", y="av_y1_4", color="cluster",
                    hover_data=[c for c in ["player_name","draft_year","team","pick","pos_group"] if c in pp.columns],
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_layout(legend_title_text="Cluster")
            else:
                st.warning("`player_clusters.csv` is missing required columns; showing position groups instead.")
                fig = px.scatter(
                    pp, x="apy_m", y="av_y1_4", color=("pos_group" if "pos_group" in pp.columns else None),
                    hover_data=[c for c in ["player_name","draft_year","team","pick","pos_group"] if c in pp.columns],
                    color_discrete_sequence=COLORWAY,
                )
                fig.update_layout(legend_title_text="Position group")
        else:
            st.warning("`player_clusters.csv` not found next to app.py — showing position groups instead.")
            fig = px.scatter(
                pp, x="apy_m", y="av_y1_4", color=("pos_group" if "pos_group" in pp.columns else None),
                hover_data=[c for c in ["player_name","draft_year","team","pick","pos_group"] if c in pp.columns],
                color_discrete_sequence=COLORWAY,
            )
            fig.update_layout(legend_title_text="Position group")
    else:
        fig = px.scatter(
            pp, x="apy_m", y="av_y1_4", color=("pos_group" if "pos_group" in pp.columns else None),
            hover_data=[c for c in ["player_name","draft_year","team","pick","pos_group"] if c in pp.columns],
            color_discrete_sequence=COLORWAY,
        )
        fig.update_layout(legend_title_text="Position group")

    fig.update_layout(
        xaxis_title="APY ($M)", yaxis_title="AV (Y1–4)",
        height=420, margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, width="stretch")

# ============= 4) Learning over time ============
from pathlib import Path

st.subheader("Learning over time — 3-year rolling Surplus (by pick band)")
st.caption("Tracks a 3-year rolling average of Surplus by pick band. Upward lines mean teams are getting better at finding value in that band; downward lines mean worse. Forecast (if shown) is exploratory only.")


if "roll_f" not in globals() or roll_f.empty or not {"draft_year","surplus_3yr"}.issubset(roll_f.columns):
    st.info("No rolling series for current selection.")
else:
    fig = px.line(
        roll_f, x="draft_year", y="surplus_3yr",
        color=("pick_band" if "pick_band" in roll_f.columns else None),
        line_group=("pick_band" if "pick_band" in roll_f.columns else None),
        color_discrete_sequence=COLORWAY
    )

    # Always show the toggle. If file missing, we warn; chart still renders.
    show_fc = st.toggle("Show forecast overlay (2022–2024)", value=False)
    fc_path = Path("learning_forecast.csv")
    if show_fc:
        if fc_path.exists():
            f = pd.read_csv(fc_path)
            if {"draft_year","surplus_3yr_forecast"}.issubset(f.columns):
                groups = (f.groupby("pick_band") if "pick_band" in f.columns else [("all", f)])
                allowed = set(roll_f["pick_band"]) if "pick_band" in roll_f.columns else None
                for band, g in groups:
                    if allowed is not None and band not in allowed:
                        continue
                    fig.add_scatter(
                        x=g["draft_year"], y=g["surplus_3yr_forecast"],
                        mode="lines", name=(f"{band} forecast" if band != "all" else "forecast"),
                        line=dict(dash="dot", width=2, color="rgba(255,255,255,0.6)")
                    )
            else:
                st.warning("`learning_forecast.csv` is missing the `surplus_3yr_forecast` column.")
        else:
            st.warning("`learning_forecast.csv` not found next to app.py — overlay skipped.")

    fig.update_layout(
        xaxis_title="Draft Year", yaxis_title="3-yr rolling Surplus",
        height=420, margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, width="stretch")

    st.caption("Solid = observed (last fully mature draft: 2021). Dotted = forecast for not-yet-mature classes (2022–2024).")

# ============ 5) Top/Bottom teams (by Surplus or ROI) =========================
st.subheader("Teams — Top / Bottom by ROI or Surplus")
st.caption('Ranks teams by the chosen metric across the current filters. Bars to the right are Top (higher), to the left are Bottom (lower). Use the slider to set how many per side.')
metric_choice = st.radio("Ranking metric", ["Surplus", "ROI"], horizontal=True, key="rank_metric")
k = st.slider("How many per side", 5, 15, 10, key="rank_k")

t = teams_f.copy()

# If these context columns aren't present, create friendly fallbacks so hover_data works
for col, default in [("era", "All"), ("pick_band", "All")]:
    if col not in t.columns:
        t[col] = default

# Choose the best available metric column for the selected ranking
candidates = {
    "Surplus": ["mean_surplus", "avg_surplus", "surplus"],
    "ROI":     ["median_roi", "roi_median", "roi"]
}
by = None
for c in candidates[metric_choice]:
    if c in t.columns:
        by = c
        break

if by is None:
    st.error(
        f"No suitable metric column found for {metric_choice}. "
        f"Tried {candidates[metric_choice]}. Available: {list(t.columns)}"
    )
else:
    tt = t.dropna(subset=[by]).copy()
    if tt.empty:
        st.info("No team rows after dropping NA for the chosen metric.")
    else:
        ranked = tt.sort_values(by, ascending=False)
        top = ranked.head(k).assign(group="Top")
        bot = ranked.tail(k).sort_values(by, ascending=True).assign(group="Bottom")
        tb = pd.concat([top, bot], ignore_index=True)

        # Keep only hover fields that actually exist
        hv = [c for c in ["era", "pick_band"] if c in tb.columns]

        fig = px.bar(
            tb, x=by, y="team", orientation="h",
            color="group",
            color_discrete_sequence=COLORWAY,
            hover_data=hv
        )
        fig.update_layout(
            xaxis_title=("Mean Surplus" if metric_choice == "Surplus" else "Median ROI"),
            yaxis_title="Team",
            height=480, margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width="stretch")

