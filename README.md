# NFL Draft Value Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nfl-draft-value.streamlit.app)

Interactive dashboard to test whether the NFL Draft behaves like an efficient market (Rounds 1–3, 2006–2024).

---

## Live App
👉 **https://nfl-draft-value.streamlit.app**

---

## What you can do
- **EV by Pick (line + ribbon):** compare expected value vs actual results within a position group.
- **Price ↔ Performance:** scatter of APY ($M) vs AV (Y1–4).
- **Learning over time:** 3-year rolling surplus trends by pick band (optional forecast overlay).
- **Surplus heatmap:** mean Surplus (AV − EV) by **Pick band × Position group**.
- **Top/Bottom teams:** rank teams by **Surplus** or **ROI** (toggle).
- **Callouts & Guide:** quick definitions, formulas, and “sweet spots / money pits”.

**Filters:** Era, Position Group, Team, Rounds (1–3), Pick bands (for Heatmap & Trend).

---

## Run Locally

```bash
# (optional) create a virtual env
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# install and run
pip install -r requirements.txt
streamlit run app.py
