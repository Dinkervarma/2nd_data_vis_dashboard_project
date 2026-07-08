"""
app_4.py — STANDALONE RUNNER FOR PAGE 4 (Industry Deep Dive)
===========================================================================
This is the file you actually run with:

    py -m streamlit run app_4.py

It loads the dataset and calls the Page 4 module (page4_industry_deep_dive.py).
Keep BOTH files in the same folder:

    your_project_folder/
    ├── app_4.py                      <- this file (run this)
    ├── page4_industry_deep_dive.py   <- the module with all the chart logic
    └── corporate_dataset.csv         <- your dataset

If you already have a full multi-page app elsewhere, don't run this file
directly — instead import render_industry_deep_dive into your main app's
router, same as before. This file exists so Page 4 works and is viewable
on its own right now.
===========================================================================
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from page4_industry_deep_dive import render_industry_deep_dive

st.set_page_config(
    page_title="AI Adoption Dashboard — Industry Deep Dive",
    page_icon="🏭",
    layout="wide",
)


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Locate the CSV automatically; fall back to a file uploader if it's
# not sitting next to this script under the expected name.
# ---------------------------------------------------------------------
DEFAULT_CSV_NAME = "corporate_dataset.csv"
csv_path = Path(__file__).parent / DEFAULT_CSV_NAME

if csv_path.exists():
    df = load_data(str(csv_path))
else:
    st.warning(
        f"Couldn't find `{DEFAULT_CSV_NAME}` next to this script. "
        f"Upload it below to continue."
    )
    uploaded = st.file_uploader("Upload corporate_dataset.csv", type="csv")
    if uploaded is None:
        st.stop()
    df = load_data(uploaded)


# ---------------------------------------------------------------------
# Render Page 4. `on_back` is left as None here since this is a
# standalone runner with no other pages to go back to — it just shows
# an info message instead. Wire this up to your real router once this
# page lives inside your full multi-page app.
# ---------------------------------------------------------------------
def _no_back_available():
    st.session_state["_back_clicked"] = True


if st.session_state.get("_back_clicked"):
    st.info("This is a standalone preview of Page 4 — there's no other page to go back to yet.")
    st.session_state["_back_clicked"] = False

render_industry_deep_dive(df, on_back=_no_back_available)
