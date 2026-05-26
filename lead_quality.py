import os, sys
import pandas as pd
from typing import Dict


def _data_dir():
    try:
        sd = os.path.dirname(os.path.abspath(__file__))
    except:
        sd = os.getcwd()
    for p in [os.path.join(sd, "data"), os.path.join(os.getcwd(), "data"), sd, os.getcwd()]:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "influencer_master.csv")):
            return p
    return sd


def _csv(name: str) -> pd.DataFrame:
    p = os.path.join(_data_dir(), name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{name} not found in {_data_dir()}")
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip().str.replace(" ", "_")
    return d


# (no decorators) helper-only functions above; Streamlit UI is launched at bottom


def count_lead_quality_by_category(df: pd.DataFrame, category_name: str = "fashion") -> Dict[str, int]:
    """Return counts of lead quality ('high','medium','low') for a category.

    - Case-insensitive match on `Category_Name`.
    - Expects `Category_Name` and `Lead_Quality` columns in `df`.
    """
    if "Category_Name" not in df.columns or "Lead_Quality" not in df.columns:
        raise ValueError("DataFrame must contain 'Category_Name' and 'Lead_Quality' columns")
    mask = df["Category_Name"].astype(str).str.lower() == str(category_name).lower()
    s = df.loc[mask, "Lead_Quality"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    return s.to_dict()


def st_show_lead_quality_counts(df: pd.DataFrame, category_name: str = "fashion") -> None:
    """Display a small table and dict of lead-quality counts in Streamlit.

    Usage in `app.py`:
        from lead_quality import st_show_lead_quality_counts
        st_show_lead_quality_counts(leads, "fashion")
    """
    import streamlit as st

    counts = count_lead_quality_by_category(df, category_name)
    st.markdown(f"**{category_name.title()} — Lead Quality Counts**")
    tbl = pd.DataFrame({"Lead_Quality": ["high", "medium", "low"],
                        "Count": [counts["high"], counts["medium"], counts["low"]]})
    st.table(tbl)
    st.write(counts)
def load_leads_from_data() -> pd.DataFrame:
    """Load influencer master from the local `data/` folder and compute Lead_Quality if missing."""
    inf = _csv("influencer_master.csv")
    # normalize some columns used by the original app
    inf = inf.rename(columns={
        "Followers": "Follower_Count",
        "Following": "Following_Count",
        "Total_Engagement": "Engagement",
        "engagement": "Engagement",
        "engagement_rate": "Engagement_Rate",
        "handle": "Handle",
        "category_id": "Category_ID",
        "category_name": "Category_Name"
    })

    # basic checks
    if "Follower_Count" not in inf.columns:
        inf["Follower_Count"] = 0
    if "Engagement_Rate" not in inf.columns:
        inf["Engagement_Rate"] = 0.0

    # compute a comparable Lead_Score and Lead_Quality if missing
    if "Lead_Score" not in inf.columns:
        er_score = inf["Engagement_Rate"].astype(float) * 0.4
        fol_score = (inf["Follower_Count"].clip(upper=2_000_000).astype(float) / 2_000_000) * 30
        raw = (er_score + fol_score).round(4)
        r_min, r_max = raw.min(), raw.max()
        # avoid division by zero
        if r_max - r_min == 0:
            inf["Lead_Score"] = 50.0
        else:
            inf["Lead_Score"] = ((raw - r_min) / (r_max - r_min) * 100).round(2)

    if "Lead_Quality" not in inf.columns:
        inf["Lead_Quality"] = pd.cut(
            inf["Lead_Score"].astype(float), bins=[-float("inf"), 30, 60, float("inf")],
            labels=["low", "medium", "high"]).astype(str)

    return inf


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Lead Quality — Fashion", layout="wide")
    st.title("Fashion — Lead Quality Overview")

    try:
        leads = load_leads_from_data()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    cats = sorted(leads["Category_Name"].dropna().unique().tolist()) if "Category_Name" in leads.columns else []
    default = "fashion" if "fashion" in [c.lower() for c in cats] else (cats[0] if cats else "fashion")
    sel = st.selectbox("Select category", options=cats or [default], index=(cats.index(default) if default in cats else 0))

    st_show_lead_quality_counts(leads, sel)


# Launch the Streamlit app when this module is executed
try:
    run_streamlit_app()
except Exception:
    # If someone runs `python lead_quality.py` without streamlit, show a helpful message
    if __name__ == "__main__":
        print("Run with: streamlit run lead_quality.py to open the Streamlit app")
