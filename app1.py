import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from streamlit_option_menu import option_menu

st.set_page_config(page_title="InfluenceIQ", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0d0f14;}
.block-container{padding:1.6rem 2.2rem 3rem;max-width:1280px;}
[data-testid="stSidebar"]{background:#0f1117;border-right:1px solid #1e2535;}
[data-testid="stSidebar"] *{color:#9ba3c4 !important;}
section[data-testid="stSidebar"] label{
  font-size:0.7rem !important;text-transform:uppercase;
  letter-spacing:.08em;color:#5c6488 !important;}
.kpi{background:#141720;border:1px solid #2a2f45;border-radius:12px;
  padding:20px 22px;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;top:0;right:0;width:52px;height:52px;
  border-radius:0 12px 0 52px;opacity:.12;background:var(--ac,#4f8ef7);}
.kpi-label{font-size:10px;color:#5c6488;font-weight:500;letter-spacing:.6px;
  text-transform:uppercase;margin-bottom:10px;}
.kpi-value{font-family:'DM Mono',monospace;font-size:28px;font-weight:600;
  line-height:1;color:var(--ac,#4f8ef7);}
.kpi-sub{font-size:11px;color:#5c6488;margin-top:6px;}
.sec{font-size:10px;font-weight:600;color:#5c6488;text-transform:uppercase;
  letter-spacing:.7px;padding-bottom:8px;border-bottom:1px solid #1c2030;
  margin-bottom:16px;margin-top:28px;}
.insight{background:#141720;border:1px solid #2a2f45;
  border-left:3px solid var(--ac,#4f8ef7);border-radius:10px;
  padding:14px 16px;font-size:13px;color:#9ba3c4;line-height:1.6;}
.insight b{color:#e8eaf6;}
.chip{display:inline-block;background:rgba(79,142,247,.12);color:#4f8ef7;
  font-size:10px;font-weight:500;padding:2px 8px;border-radius:20px;
  border:1px solid rgba(79,142,247,.25);margin:2px 3px 2px 0;}
.pb-wrap{margin-bottom:12px;}
.pb-row{display:flex;justify-content:space-between;font-size:12px;
  color:#9ba3c4;margin-bottom:4px;}
.pb-val{font-family:'DM Mono',monospace;font-size:12px;}
.pb-track{height:5px;background:#1c2030;border-radius:3px;overflow:hidden;}
.pb-fill{height:100%;border-radius:3px;}
.post-banner{background:#141720;border:1px solid #2a2f45;border-radius:14px;
  padding:22px 26px;margin-bottom:18px;}
.post-banner-id{font-family:'DM Mono',monospace;font-size:11px;color:#4f8ef7;
  letter-spacing:.05em;margin-bottom:4px;}
.post-banner-handle{font-size:1.1rem;font-weight:600;color:#e8eaf6;margin-bottom:2px;}
.post-banner-meta{font-size:12px;color:#5c6488;}
.stat-pill{display:inline-flex;align-items:center;gap:6px;background:#1c2030;
  border:1px solid #2a2f45;border-radius:8px;padding:6px 12px;
  font-size:12px;color:#9ba3c4;margin:3px 4px 3px 0;}
.stat-pill b{color:var(--pc,#4f8ef7);font-family:'DM Mono',monospace;}
.about-hero{background:linear-gradient(135deg,#141720 0%,#0f1825 100%);
  border:1px solid #2a2f45;border-radius:16px;padding:32px 36px;margin-bottom:20px;}
.about-card{background:#141720;border:1px solid #2a2f45;border-radius:14px;
  padding:24px 26px;height:100%;}
.about-card-green{border-left:4px solid #34d399;}
.about-card-blue {border-left:4px solid #4f8ef7;}
.about-card-amber{border-left:4px solid #fbbf24;}
.about-card-pink {border-left:4px solid #f472b6;}
.about-card-purple{border-left:4px solid #a78bfa;}
.about-title{font-size:13px;font-weight:600;color:#e8eaf6;margin-bottom:12px;
  display:flex;align-items:center;gap:8px;}
.about-body{font-size:13px;color:#9ba3c4;line-height:1.7;}
.about-tag{display:inline-block;font-family:'DM Mono',monospace;font-size:11px;
  background:#1c2030;color:#a3c4fd;padding:1px 7px;border-radius:4px;
  border:1px solid #2a2f45;margin:1px 2px;}
.about-li{display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;
  font-size:13px;color:#9ba3c4;line-height:1.6;}
.about-li-dot{width:6px;height:6px;border-radius:50%;margin-top:6px;
  flex-shrink:0;background:var(--dot,#4f8ef7);}
.about-divider{border:none;border-top:1px solid #1c2030;margin:8px 0 16px;}
[data-testid="stDataFrame"] *{
  font-family:'DM Mono',monospace !important;font-size:12px !important;}
</style>
""", unsafe_allow_html=True)

CAT_CLR = {"tech":"#4f8ef7","fashion":"#f472b6","fitness":"#34d399",
           "travel":"#a78bfa","food":"#fbbf24"}
Q_CLR   = {"high":"#34d399","medium":"#fbbf24","low":"#f87171"}
DL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans",color="#9ba3c4",size=12),
    xaxis=dict(gridcolor="#1c2030",linecolor="#2a2f45",zeroline=False),
    yaxis=dict(gridcolor="#1c2030",linecolor="#2a2f45",zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="#2a2f45",font=dict(size=11)),
    margin=dict(l=24,r=24,t=40,b=24))

def dark(fig, h=300):
    fig.update_layout(**DL, height=h)
    return fig

# ── DATA LOADING ──────────────────────────────────────────
def _data_dir():
    try:    sd = os.path.dirname(os.path.abspath(__file__))
    except: sd = os.path.dirname(os.path.abspath(sys.argv[0]))
    for p in [os.path.join(sd,"data"), os.path.join(os.getcwd(),"data"), sd, os.getcwd()]:
        if os.path.isdir(p) and any(os.path.exists(os.path.join(p,f))
           for f in ["influencer_master.csv","post_metrics.csv"]):
            return p
    return sd

def _csv(name):
    p = os.path.join(_data_dir(), name)
    if not os.path.exists(p):
        st.error(f"❌ `{name}` not found in `{_data_dir()}`."); st.stop()
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip().str.replace(" ","_")
    return d

@st.cache_data(show_spinner="Loading data…")
def load_data():
    inf  = _csv("influencer_master.csv")
    post = _csv("post_metrics.csv")
    cat  = _csv("category_dim.csv")

    inf  = inf.rename(columns={"Followers":"Follower_Count","Following":"Following_Count",
                                "Total_Engagement":"Engagement","engagement":"Engagement",
                                "engagement_rate":"Engagement_Rate","handle":"Handle",
                                "category_id":"Category_ID"})
    post = post.rename(columns={"Total_Engagement":"Engagement","engagement":"Engagement",
                                 "post_date":"Post_Date","likes":"Likes","comments":"Comments",
                                 "handle":"Handle","sentiment_score":"Sentiment_Score",
                                 "hashtags":"Hashtags","post_id":"Post_ID"})
    if "Post_ID" not in post.columns:
        post["Post_ID"] = ["POST_" + str(i+1).zfill(6) for i in range(len(post))]
    cat  = cat.rename(columns={"saas_relevance":"SaaS_Relevance_Weight",
                                "SaaS_Relevance":"SaaS_Relevance_Weight",
                                "category_id":"Category_ID","category_name":"Category_Name"})

    if "Category_ID" in inf.columns and "Category_ID" in cat.columns:
        inf = inf.merge(cat, on="Category_ID", how="left")

    inf["SaaS_Relevance"] = inf["SaaS_Relevance_Weight"].fillna(0.5) \
                            if "SaaS_Relevance_Weight" in inf.columns else 0.5

    for c in ["Handle","Follower_Count","Engagement_Rate","Engagement"]:
        if c not in inf.columns:
            st.error(f"Missing column `{c}`"); st.stop()

    post["Post_Date"] = pd.to_datetime(post["Post_Date"], dayfirst=True, errors="coerce")
    post["Month"]     = post["Post_Date"].dt.to_period("M").dt.to_timestamp()
    post["MonthName"] = post["Post_Date"].dt.strftime("%b %Y")
    post["Year"]      = post["Post_Date"].dt.year
    if "Sentiment_Score" in post.columns:
        post["Sentiment_Bin"] = pd.cut(post["Sentiment_Score"],
            bins=[-1.1,-0.1,0.1,1.1], labels=["Negative","Neutral","Positive"])

    if "Sentiment_Score" in post.columns:
        avg_s = post.groupby("Handle")["Sentiment_Score"].mean().rename("Avg_Sentiment")
        inf   = inf.merge(avg_s, on="Handle", how="left")
    inf["Avg_Sentiment"] = inf.get("Avg_Sentiment", pd.Series(0,index=inf.index)).fillna(0)

    er_score   = inf["Engagement_Rate"] * 0.4
    fol_score  = (inf["Follower_Count"].clip(upper=2_000_000) / 2_000_000) * 30
    sent_score = ((inf["Avg_Sentiment"] + 1) / 2) * 20
    saas_score = inf["SaaS_Relevance"] * 10
    raw = (er_score + fol_score + sent_score + saas_score).round(4)
    r_min, r_max = raw.min(), raw.max()
    inf["Lead_Score"] = ((raw - r_min) / (r_max - r_min) * 100).round(2)

    inf["Lead_Quality"] = pd.cut(
        inf["Lead_Score"], bins=[-np.inf, 30, 60, np.inf],
        labels=["low","medium","high"]).astype(str)

    def _tier(f):
        if f < 10_000:    return "Nano (<10K)"
        if f < 100_000:   return "Micro (10K-100K)"
        if f < 500_000:   return "Mid (100K-500K)"
        if f < 1_000_000: return "Macro (500K-1M)"
        return "Mega (1M+)"

    inf["Follower_Tier"] = inf["Follower_Count"].apply(_tier)
    fc = inf.get("Following_Count", pd.Series(0,index=inf.index))
    inf["FF_Ratio"] = (fc / inf["Follower_Count"].replace(0,np.nan)).round(4).fillna(0)
    inf["Action"]   = inf["Lead_Quality"].map(
        {"high":"🔥 Contact Now","medium":"⏳ Nurture","low":"🗑 Ignore"})

    mc   = [c for c in ["Handle","Category_Name","Lead_Quality","Lead_Score"] if c in inf.columns]
    post = post.merge(inf[mc], on="Handle", how="left")
    return inf, post

leads_full, posts_full = load_data()

# ── HELPERS ───────────────────────────────────────────────
def fmt(n):
    try: n = float(n)
    except: return str(n)
    if np.isnan(n): return "—"
    if n>=1e9: return f"{n/1e9:.1f}B"
    if n>=1e6: return f"{n/1e6:.1f}M"
    if n>=1e3: return f"{n/1e3:.1f}K"
    return f"{n:.1f}"

def safe_int(v):
    try:
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default

def kpi(label, value, sub="", ac="#4f8ef7"):
    return (f'<div class="kpi" style="--ac:{ac}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def pb(label, val_s, pct, color):
    return (f'<div class="pb-wrap"><div class="pb-row">'
            f'<span>{label}</span>'
            f'<span class="pb-val" style="color:{color}">{val_s}</span></div>'
            f'<div class="pb-track">'
            f'<div class="pb-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div>'
            f'</div></div>')

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:.8rem 0 1.2rem">'
        '<div style="font-size:1.25rem;font-weight:700;color:#e8eaf6">'
        '⚡ Influence<span style="color:#4f8ef7;font-weight:300">IQ</span></div>'
        '<div style="font-size:10px;color:#3a4060;text-transform:uppercase;'
        'letter-spacing:.1em;margin-top:2px">AI Influencer Intelligence</div></div>',
        unsafe_allow_html=True)

    st.markdown("**🎛 Filters**")
    st.markdown(
        '<div style="font-size:10px;color:#3a4060;margin:-6px 0 10px;line-height:1.55">'
        'Empty = all data shown. Pick values to filter.</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    all_cats  = sorted(leads_full["Category_Name"].dropna().unique()) \
                if "Category_Name" in leads_full.columns else []
    sel_cats  = st.multiselect("📂 Category", all_cats, default=[],
                                placeholder="All categories")
    sel_qual  = st.multiselect("🎯 Lead Quality", ["high","medium","low"], default=[],
                                placeholder="All quality tiers")

    er_lo = float(leads_full["Engagement_Rate"].min())
    er_hi = float(leads_full["Engagement_Rate"].max())
    sel_er = st.slider("📈 Engagement Rate (%)", er_lo, er_hi, (er_lo, er_hi), step=0.1)

    fo_lo = int(leads_full["Follower_Count"].min())
    fo_hi = int(leads_full["Follower_Count"].max())
    sel_fo = st.slider("👥 Followers", fo_lo, fo_hi, (fo_lo, fo_hi), step=10_000)

    sel_score = st.slider("🤖 Min Lead Score", 0, 100, 0)
    st.markdown("---")

    all_tiers = sorted(leads_full["Follower_Tier"].unique())
    sel_tiers = st.multiselect("🏷 Follower Tier", all_tiers, default=[],
                                placeholder="All tiers")

    d_lo = posts_full["Post_Date"].dropna().min().date()
    d_hi = posts_full["Post_Date"].dropna().max().date()
    sel_dates = st.date_input("📅 Date Range", (d_lo, d_hi))

    all_yrs = sorted(posts_full["Year"].dropna().astype(int).unique())
    sel_yrs = st.multiselect("📅 Year", all_yrs, default=[],
                              placeholder="All years")

    st.markdown("---")
    if st.button("↺ Reset Filters", use_container_width=True):
        st.rerun()

# ── FILTER FUNCTIONS ──────────────────────────────────────
def fl(d):
    if sel_cats and "Category_Name" in d.columns:
        d = d[d["Category_Name"].isin(sel_cats)]
    if sel_qual:
        d = d[d["Lead_Quality"].isin(sel_qual)]
    d = d[d["Engagement_Rate"].between(*sel_er)]
    d = d[d["Follower_Count"].between(*sel_fo)]
    if sel_tiers: d = d[d["Follower_Tier"].isin(sel_tiers)]
    return d[d["Lead_Score"] >= sel_score]

def fp(d):
    if len(sel_dates)==2:
        d = d[(d["Post_Date"].dt.date >= sel_dates[0]) &
              (d["Post_Date"].dt.date <= sel_dates[1])]
    if sel_yrs:  d = d[d["Year"].isin(sel_yrs)]
    if sel_cats and "Category_Name" in d.columns:
        d = d[d["Category_Name"].isin(sel_cats)]
    if sel_qual and "Lead_Quality" in d.columns:
        d = d[d["Lead_Quality"].isin(sel_qual)]
    return d

leads = fl(leads_full.copy())
posts = fp(posts_full.copy())

is_filtered = (bool(sel_cats) or bool(sel_qual) or bool(sel_tiers) or bool(sel_yrs)
               or sel_er!=(er_lo,er_hi) or sel_fo!=(fo_lo,fo_hi) or sel_score>0)

# ── HEADER ────────────────────────────────────────────────
chips = "".join(
    f'<span class="chip">{v}</span>'
    for v in list(sel_cats)+list(sel_qual)+list(sel_tiers)+list(sel_yrs))

st.markdown(
    f'<div style="background:#141720;border:1px solid #2a2f45;border-radius:12px;'
    f'padding:12px 20px 10px;margin-bottom:1.2rem;">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;'
    f'flex-wrap:wrap;gap:8px;">'
    f'<div><span style="font-size:1.4rem;font-weight:700;color:#e8eaf6">⚡ InfluenceIQ</span>'
    f'<span style="font-size:.8rem;color:#5c6488;margin-left:10px">'
    f'AI Influencer Intelligence Dashboard</span></div>'
    f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:#3a4060;'
    f'border:1px solid #1c2030;padding:2px 10px;border-radius:20px;">'
    f'{"🔍 " if is_filtered else "📊 "}{len(leads):,} / {len(leads_full):,} records</span>'
    f'</div>'
    f'{"<div style=margin-top:6px>"+chips+"</div>" if chips else ""}'
    f'</div>',
    unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────
page = option_menu(None,
    ["Executive Overview","Lead Intelligence",
     "Post Analytics","AI Lead Scoring","About"],
    icons=["bar-chart-fill","people-fill","chat-dots-fill","robot","info-circle-fill"],
    orientation="horizontal",
    styles={
        "container":         {"background-color":"#141720","border":"1px solid #2a2f45",
                              "border-radius":"10px","padding":"5px 10px"},
        "nav-link":          {"font-size":"13px","color":"#9ba3c4",
                              "border-radius":"8px","padding":"7px 14px"},
        "nav-link-selected": {"background-color":"#1a3f8f","color":"#a3c4fd","font-weight":"500"},
        "icon":              {"font-size":"13px"},
    })

# ==========================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==========================================================
if page == "Executive Overview":

    total  = len(leads)
    t_eng  = leads["Engagement"].sum()
    avg_er = leads["Engagement_Rate"].mean() if total else 0
    hq     = (leads["Lead_Quality"]=="high").sum()
    n_cats = leads["Category_Name"].nunique() if "Category_Name" in leads.columns else 0
    eqr    = (leads["Engagement"] / leads["Follower_Count"].replace(0,np.nan)).mean()
    eqr    = 0 if (eqr is None or (isinstance(eqr, float) and np.isnan(eqr))) else eqr

    scope = "full dataset" if not is_filtered else f"{total:,} filtered records"
    st.markdown(
        f'<p style="color:#5c6488;font-size:13px;margin-bottom:1rem">'
        f'High-level view of the influencer ecosystem · '
        f'<b style="color:#4f8ef7">{scope}</b></p>',
        unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("Total Influencers",   fmt(total),
                    f"{n_cats} {'category' if n_cats==1 else 'categories'}",  "#4f8ef7"), unsafe_allow_html=True)
    k2.markdown(kpi("Total Engagement",    fmt(t_eng),
                    "across selected handles",                                  "#34d399"), unsafe_allow_html=True)
    k3.markdown(kpi("Avg Engagement Rate", f"{avg_er:.2f}%",
                    "average across selection",                                 "#fbbf24"), unsafe_allow_html=True)
    k4.markdown(kpi("Engagement Quality",  f"{eqr:.4f}",
                    "avg eng ÷ followers (higher = better)",                   "#a78bfa"), unsafe_allow_html=True)

    sec("🧠 Smart Insights")
    if total > 0 and "Category_Name" in leads.columns:
        top_er   = leads.groupby("Category_Name")["Engagement_Rate"].mean().idxmax()
        top_fol  = leads.groupby("Category_Name")["Follower_Count"].mean().idxmax()
        med      = (leads["Lead_Quality"]=="medium").sum()
        top_hq   = (leads[leads["Lead_Quality"]=="high"]
                    .groupby("Category_Name").size().idxmax() if hq else "—")
        avg_sc   = leads["Lead_Score"].mean()
        tier_top = leads["Follower_Tier"].value_counts().idxmax()
        top_eqr  = (leads.assign(EQR=leads["Engagement"]/leads["Follower_Count"].replace(0,np.nan))
                    .groupby("Category_Name")["EQR"].mean().idxmax())

        i1, i2, i3 = st.columns(3)
        i1.markdown(
            f'<div class="insight" style="--ac:#4f8ef7">'
            f'📌 <b>{top_er.title()}</b> leads with the highest avg engagement rate '
            f'across {total:,} influencers in view.</div>', unsafe_allow_html=True)
        i2.markdown(
            f'<div class="insight" style="--ac:#34d399">'
            f'📡 <b>{top_fol.title()}</b> dominates follower reach — '
            f'best for brand awareness campaigns.</div>', unsafe_allow_html=True)
        i3.markdown(
            f'<div class="insight" style="--ac:#fbbf24">'
            f'🎯 <b>{hq:,}</b> leads outreach-ready '
            f'({hq/max(total,1)*100:.1f}%). '
            f'Top vertical: <b>{str(top_hq).title()}</b>.</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        i4, i5, i6 = st.columns(3)
        i4.markdown(
            f'<div class="insight" style="--ac:#a78bfa">'
            f'🏷 Dominant tier: <b>{tier_top}</b>. '
            f'Avg lead score: <b>{avg_sc:.1f} / 100</b>.</div>', unsafe_allow_html=True)
        i5.markdown(
            f'<div class="insight" style="--ac:#f472b6">'
            f'💡 <b>{top_eqr.title()}</b> has the best engagement quality ratio — '
            f'most impactful audience per follower.</div>', unsafe_allow_html=True)
        i6.markdown(
            f'<div class="insight" style="--ac:#34d399">'
            f'⏳ <b>{med:,}</b> medium leads '
            f'({med/max(total,1)*100:.1f}%) in the nurture pipeline.</div>',
            unsafe_allow_html=True)
    else:
        st.info("No data matches the current filters — adjust the sidebar.")

    sec("Engagement Trend & Lead Pipeline")
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        monthly = (posts.groupby("Month")["Engagement"].sum()
                   .reset_index().sort_values("Month"))
        fig = go.Figure(go.Scatter(
            x=monthly["Month"], y=monthly["Engagement"],
            mode="lines+markers",
            line=dict(color="#4f8ef7", width=2.5),
            marker=dict(size=5, color="#4f8ef7"),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.07)"))
        dark(fig, 300)
        fig.update_layout(title=dict(text="Monthly Engagement Trend",
                                     font=dict(size=12,color="#5c6488")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        qc = leads["Lead_Quality"].value_counts().reset_index()
        qc.columns = ["Quality", "Count"]
        order = ["high","medium","low"]
        qc["_ord"] = qc["Quality"].map({v:i for i,v in enumerate(order)})
        qc = qc.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
        fig2 = go.Figure(go.Pie(
            labels=qc["Quality"], values=qc["Count"], hole=0.62,
            marker=dict(colors=[Q_CLR.get(q,"#888") for q in qc["Quality"]],
                        line=dict(color="#0d0f14",width=2)),
            texttemplate="%{label}<br>%{percent:.1%}",
            textfont=dict(size=11,color="#e8eaf6"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent:.2%}<extra></extra>",
            direction="clockwise", sort=False))
        dark(fig2, 300)
        fig2.update_layout(
            title=dict(text="Lead Quality Distribution",font=dict(size=12,color="#5c6488")),
            legend=dict(orientation="h",y=-0.18,font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    sec("Engagement Quality by Category")
    if "Category_Name" in leads.columns:
        eqr_cat = (leads.assign(EQR=leads["Engagement"]/leads["Follower_Count"].replace(0,np.nan))
                   .groupby("Category_Name")["EQR"].mean()
                   .reset_index().sort_values("EQR", ascending=False))
        fig3 = go.Figure()
        for _, row in eqr_cat.iterrows():
            cat = str(row["Category_Name"]).lower()
            color = CAT_CLR.get(cat, "#4f8ef7")
            fig3.add_trace(go.Bar(
                x=[row["Category_Name"]], y=[row["EQR"]],
                name=row["Category_Name"],
                marker_color=color, marker_line_width=0,
                text=[f"{row['EQR']:.4f}"], textposition="outside",
                textfont=dict(color="#9ba3c4", size=11),
            ))
        dark(fig3, 260)
        fig3.update_layout(
            title=dict(text="Avg Engagement ÷ Followers per Category (higher = better audience quality)",
                       font=dict(size=12,color="#5c6488")),
            showlegend=False, bargap=0.35)
        st.plotly_chart(fig3, use_container_width=True)


# ==========================================================
# PAGE 2 — LEAD INTELLIGENCE
# ==========================================================
elif page == "Lead Intelligence":
    total  = len(leads)
    hq     = (leads["Lead_Quality"]=="high").sum()
    avg_ff = leads["FF_Ratio"].mean()
    hi_er  = (leads["Engagement_Rate"] > 15).sum()

    st.markdown(
        f'<p style="color:#5c6488;font-size:13px;margin-bottom:1rem">'
        f'Influencer quality signals, authenticity analysis, and profile ranking · '
        f'<b style="color:#4f8ef7">{total:,} records in view</b></p>',
        unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("In View",      fmt(total),       "current filter",                     "#4f8ef7"), unsafe_allow_html=True)
    k2.markdown(kpi("High-Quality", fmt(hq),          f"{hq/max(total,1)*100:.1f}% of view","#34d399"), unsafe_allow_html=True)
    k3.markdown(kpi("Avg FF Ratio", f"{avg_ff:.3f}",  "lower = more authentic",              "#fbbf24"), unsafe_allow_html=True)
    k4.markdown(kpi("ER > 15%",     fmt(hi_er),       "immediate outreach targets",          "#f472b6"), unsafe_allow_html=True)

    sec("Follower Reach vs Engagement Rate")
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        samp = leads.sample(min(2000,len(leads)), random_state=42)
        kws  = dict(opacity=0.55)
        if "Category_Name" in samp.columns:
            kws["color"] = "Category_Name"; kws["color_discrete_map"] = CAT_CLR
        if "Engagement" in samp.columns: kws["size"] = "Engagement"
        if "Handle"     in samp.columns: kws["hover_data"] = ["Handle","Lead_Quality"]
        fig = px.scatter(samp, x="Follower_Count", y="Engagement_Rate", **kws)
        fig.update_traces(marker_line_width=0)
        dark(fig, 320)
        fig.update_layout(title=dict(text="Followers vs Engagement Rate",
                                     font=dict(size=12,color="#5c6488")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        sec("Authenticity — Avg FF Ratio by Category")
        if "Category_Name" in leads.columns:
            ff = leads.groupby("Category_Name")["FF_Ratio"].mean().sort_values()
            max_ff = ff.max() if ff.max()>0 else 1
            html = ('<div style="background:#141720;border:1px solid #2a2f45;'
                    'border-radius:12px;padding:18px 20px">')
            for cat,val in ff.items():
                color = CAT_CLR.get(str(cat).lower(),"#4f8ef7")
                html += pb(str(cat).title(), f"{val:.3f}", val/max_ff*100, color)
            html += ('<div style="font-size:10.5px;color:#5c6488;margin-top:10px">'
                     'Lower ratio = stronger authenticity signal</div></div>')
            st.markdown(html, unsafe_allow_html=True)

    sec("Follower Tier Distribution & Engagement Box")
    ch3, ch4 = st.columns(2)

    with ch3:
        if "Category_Name" in leads.columns:
            tier_order = ["Nano (<10K)","Micro (10K-100K)","Mid (100K-500K)","Macro (500K-1M)","Mega (1M+)"]
            td = leads.groupby(["Category_Name","Follower_Tier"]).size().reset_index(name="Count")
            fig3 = px.bar(td, x="Category_Name", y="Count", color="Follower_Tier",
                          barmode="stack", category_orders={"Follower_Tier":tier_order},
                          color_discrete_sequence=["#1e3a5f","#1d4ed8","#4f8ef7","#93c5fd","#dbeafe"])
            fig3.update_traces(marker_line_width=0)
            dark(fig3, 290)
            fig3.update_layout(
                title=dict(text="Follower Tier Stack by Category",font=dict(size=12,color="#5c6488")),
                bargap=0.25)
            st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        if "Category_Name" in leads.columns:
            fig4 = px.box(leads, x="Category_Name", y="Engagement_Rate",
                          color="Category_Name", color_discrete_map=CAT_CLR, points=False)
            fig4.update_traces(line_width=1.5)
            dark(fig4, 290)
            fig4.update_layout(
                title=dict(text="Engagement Rate Distribution by Category",font=dict(size=12,color="#5c6488")),
                showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    sec("Top Influencer Profiles")
    s1, s2 = st.columns([2,1])
    sort_by = s1.selectbox("Sort by",
        ["Lead_Score","Follower_Count","Engagement_Rate","Engagement"], key="li_s")
    n_show  = s2.selectbox("Show", [20,50,100], key="li_n")

    sc  = [c for c in ["Handle","Category_Name","Follower_Count","Engagement_Rate",
                        "Lead_Score","Lead_Quality","Action"] if c in leads.columns]
    tbl = leads.nlargest(n_show, sort_by)[sc].copy()
    if "Follower_Count"  in tbl.columns: tbl["Follower_Count"]  = tbl["Follower_Count"].apply(fmt)
    if "Engagement_Rate" in tbl.columns: tbl["Engagement_Rate"] = tbl["Engagement_Rate"].round(2)
    if "Lead_Score"      in tbl.columns: tbl["Lead_Score"]      = tbl["Lead_Score"].round(1)
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=320)


# ==========================================================
# PAGE 3 — POST ANALYTICS  (merged with Post Tracking)
# PART A: overall charts  |  PART B: post inspector
# ==========================================================
elif page == "Post Analytics":

    avg_l  = posts["Likes"].mean()    if "Likes"    in posts.columns else 0
    avg_c  = posts["Comments"].mean() if "Comments" in posts.columns else 0
    avg_l  = 0 if (avg_l is None or (isinstance(avg_l, float) and np.isnan(avg_l))) else avg_l
    avg_c  = 0 if (avg_c is None or (isinstance(avg_c, float) and np.isnan(avg_c))) else avg_c
    eqr_post = (posts["Engagement"] / posts["Likes"].replace(0,np.nan)).mean() \
               if "Likes" in posts.columns else 0
    eqr_post = 0 if (eqr_post is None or (isinstance(eqr_post, float) and np.isnan(eqr_post))) else eqr_post
    peak_m   = posts.groupby("MonthName")["Engagement"].sum().idxmax() if len(posts) else "—"

    st.markdown(
        f'<p style="color:#5c6488;font-size:13px;margin-bottom:1rem">'
        f'Content performance, engagement patterns, and post inspector · '
        f'<b style="color:#4f8ef7">{len(posts):,} posts in view</b></p>',
        unsafe_allow_html=True)

    if len(posts) == 0:
        st.info("No posts match the current filters.")
        st.stop()
# ════════════════════════════════════════════════════════
    # PART B — POST INSPECTOR (merged from Post Tracking)
    # ════════════════════════════════════════════════════════
    st.markdown(
        '<div style="height:2px;background:linear-gradient(90deg,#1a3f8f,#4f8ef7,transparent);'
        'border-radius:2px;margin:32px 0 8px;"></div>',
        unsafe_allow_html=True)
    sec("🔍 Post Inspector — Search & Drill Down")

    st.markdown(
        '<p style="color:#5c6488;font-size:12.5px;margin:-8px 0 16px">'
        'Search any Post ID or Handle to view full metrics, engagement history, '
        'and influencer profile.</p>',
        unsafe_allow_html=True)

    # Search + filter row
    src1, src2, src3 = st.columns([3,2,2])
    search_text = src1.text_input("🔍 Search Post ID or Handle",
                                   placeholder="e.g. POST000001 or @username",
                                   key="pt_search")
    month_opts  = ["All Months"] + sorted(
        posts_full["MonthName"].dropna().unique().tolist(),
        key=lambda x: pd.to_datetime(x, format="%b %Y"))
    sel_month   = src2.selectbox("📅 Month", month_opts, key="pt_month")
    cat_opts_pt = ["All Categories"] + sorted(
        posts_full["Category_Name"].dropna().unique().tolist()
        if "Category_Name" in posts_full.columns else [])
    sel_cat_pt  = src3.selectbox("📂 Category", cat_opts_pt, key="pt_cat")

    # Build search pool from full dataset (not filtered) so any post is findable
    pool = posts_full.copy()
    if sel_month != "All Months":
        pool = pool[pool["MonthName"] == sel_month]
    if sel_cat_pt != "All Categories" and "Category_Name" in pool.columns:
        pool = pool[pool["Category_Name"] == sel_cat_pt]
    if search_text.strip():
        q = search_text.strip().lower()
        mask = pd.Series([False]*len(pool), index=pool.index)
        if "Post_ID" in pool.columns:
            mask |= pool["Post_ID"].astype(str).str.lower().str.contains(q, na=False)
        if "Handle" in pool.columns:
            mask |= pool["Handle"].astype(str).str.lower().str.contains(q, na=False)
        pool = pool[mask]

    st.markdown(
        f'<div style="font-size:12px;color:#5c6488;margin-bottom:12px;">'
        f'{len(pool):,} post(s) match · select a Post ID below to inspect</div>',
        unsafe_allow_html=True)

    if len(pool) == 0:
        st.info("No posts match — try a different search term, month, or category.")
    else:
        post_id_col = "Post_ID" if "Post_ID" in pool.columns else pool.columns[0]
        post_ids    = pool[post_id_col].astype(str).tolist()
        sel_pid     = st.selectbox("📋 Select Post ID", post_ids, key="pt_pid")
        row = pool[pool[post_id_col].astype(str) == sel_pid].iloc[0]

        # ── Banner ─────────────────────────────────────────
        handle    = str(row.get("Handle","—"))
        post_date = str(row.get("Post_Date",""))[:10]
        category  = str(row.get("Category_Name","—")).title()
        quality   = str(row.get("Lead_Quality","—"))
        q_color   = Q_CLR.get(quality,"#9ba3c4")
        q_badge   = (f'<span style="background:{q_color}22;color:{q_color};font-size:11px;'
                     f'font-weight:500;padding:2px 9px;border-radius:20px;'
                     f'border:1px solid {q_color}44">{quality.title()}</span>')
        cat_color = CAT_CLR.get(str(row.get("Category_Name","")).lower(),"#4f8ef7")

        st.markdown(f"""
        <div class="post-banner">
          <div class="post-banner-id">{sel_pid}</div>
          <div class="post-banner-handle">@{handle}</div>
          <div class="post-banner-meta">
            {post_date} &nbsp;·&nbsp;
            <span style="color:{cat_color}">{category}</span>
            &nbsp;·&nbsp; Lead Quality: {q_badge}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Stat pills ─────────────────────────────────────
        pills_html = '<div style="margin-bottom:18px;display:flex;flex-wrap:wrap;">'
        stats = [
            ("Likes",      fmt(safe_int(row.get("Likes",0))),           "#34d399"),
            ("Comments",   fmt(safe_int(row.get("Comments",0))),         "#4f8ef7"),
            ("Engagement", fmt(safe_int(row.get("Engagement",0))),       "#a78bfa"),
            ("Lead Score", f"{safe_float(row.get('Lead_Score',0)):.1f}", "#fbbf24"),
        ]
        if handle in leads_full["Handle"].values:
            inf_row = leads_full[leads_full["Handle"]==handle].iloc[0]
            stats += [
                ("Followers", fmt(safe_int(inf_row["Follower_Count"])),               "#f472b6"),
                ("ER %",      f"{safe_float(inf_row['Engagement_Rate']):.2f}%",        "#f87171"),
            ]
        for lbl, val, color in stats:
            pills_html += (f'<span class="stat-pill" style="--pc:{color}">'
                           f'<b>{val}</b> {lbl}</span>')
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── Handle history charts ──────────────────────────
        handle_posts = posts_full[posts_full["Handle"] == handle].copy()
        handle_posts = handle_posts.sort_values("Post_Date")

        if len(handle_posts) > 1:
            sec(f"@{handle} — Engagement History")
            hc1, hc2 = st.columns(2)

            with hc1:
                fig_h1 = go.Figure()
                fig_h1.add_trace(go.Scatter(
                    x=handle_posts["Post_Date"], y=handle_posts["Engagement"],
                    mode="lines+markers",
                    line=dict(color="#4f8ef7", width=2),
                    marker=dict(size=5, color="#4f8ef7"),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.07)",
                    name="Engagement"))
                sel_date = row.get("Post_Date")
                sel_eng  = row.get("Engagement", 0)
                if pd.notna(sel_date):
                    fig_h1.add_trace(go.Scatter(
                        x=[sel_date], y=[sel_eng],
                        mode="markers",
                        marker=dict(size=13, color="#f87171", symbol="star"),
                        name="Selected post"))
                dark(fig_h1, 280)
                fig_h1.update_layout(title=dict(
                    text="Engagement Over Time  (★ = selected post)",
                    font=dict(size=12,color="#5c6488")))
                st.plotly_chart(fig_h1, use_container_width=True)

            with hc2:
                # ── NEW: Likes vs Comments for this handle
                # Horizontal bar comparison — shows each post side by side
                if "Likes" in handle_posts.columns and "Comments" in handle_posts.columns:
                    hp = handle_posts.copy()
                    hp["Label"] = hp["Post_Date"].dt.strftime("%b %d")
                    # keep last 15 posts for readability
                    hp = hp.tail(15).reset_index(drop=True)

                    fig_h2 = go.Figure()
                    fig_h2.add_trace(go.Bar(
                        y=hp["Label"], x=hp["Likes"],
                        name="Likes", orientation="h",
                        marker_color="#4f8ef7", marker_line_width=0, opacity=0.85))
                    fig_h2.add_trace(go.Bar(
                        y=hp["Label"], x=hp["Comments"],
                        name="Comments", orientation="h",
                        marker_color="#f472b6", marker_line_width=0, opacity=0.85))
                    # highlight selected post date
                    sel_label = pd.to_datetime(sel_date).strftime("%b %d") if pd.notna(sel_date) else None
                    if sel_label and sel_label in hp["Label"].values:
                        idx = hp[hp["Label"]==sel_label].index[0]
                        fig_h2.add_shape(type="rect",
                            xref="paper", yref="y",
                            x0=0, x1=1,
                            y0=idx-0.5, y1=idx+0.5,
                            fillcolor="rgba(248,113,113,0.08)",
                            line=dict(color="#f87171", width=1, dash="dot"))
                    dark(fig_h2, 280)
                    fig_h2.update_layout(
                        title=dict(text="Likes vs Comments — last 15 posts (★ = selected)",
                                   font=dict(size=12,color="#5c6488")),
                        barmode="group", bargap=0.22,
                        xaxis_title="Count",
                        legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_h2, use_container_width=True)

            # Monthly engagement bar for this handle
            sec(f"Monthly Engagement — @{handle}")
            monthly_h = handle_posts.groupby("Month")["Engagement"].sum().reset_index().sort_values("Month")
            fig_h3 = px.bar(monthly_h, x="Month", y="Engagement",
                            color_discrete_sequence=["#34d399"])
            fig_h3.update_traces(marker_line_width=0)
            dark(fig_h3, 240)
            fig_h3.update_layout(title=dict(text="Monthly Engagement for this Influencer",
                                             font=dict(size=12,color="#5c6488")), bargap=0.25)
            st.plotly_chart(fig_h3, use_container_width=True)
        else:
            st.info(f"Only one post found for @{handle} in the dataset.")

        # ── Influencer profile KPIs ────────────────────────
        sec("Influencer Profile")
        if handle in leads_full["Handle"].values:
            inf_r = leads_full[leads_full["Handle"]==handle].iloc[0]
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.markdown(kpi("Followers",       fmt(safe_int(inf_r.get("Follower_Count",0))),            "",              "#4f8ef7"), unsafe_allow_html=True)
            ic2.markdown(kpi("Engagement Rate", f"{safe_float(inf_r.get('Engagement_Rate',0)):.2f}%",   "",              "#34d399"), unsafe_allow_html=True)
            ic3.markdown(kpi("Lead Score",      f"{safe_float(inf_r.get('Lead_Score',0)):.1f}",          "",    "#fbbf24"), unsafe_allow_html=True)
            ic4.markdown(kpi("FF Ratio",        f"{safe_float(inf_r.get('FF_Ratio',0)):.3f}",            "",  "#a78bfa"), unsafe_allow_html=True)


    # ── KPIs ────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Total Posts",       fmt(len(posts)),        "in current view",               "#4f8ef7"), unsafe_allow_html=True)
    k2.markdown(kpi("Avg Likes",         fmt(safe_int(avg_l)),   "per post",                      "#34d399"), unsafe_allow_html=True)
    k3.markdown(kpi("Avg Comments",      fmt(safe_int(avg_c)),   "per post",                      "#a78bfa"), unsafe_allow_html=True)
    k4.markdown(kpi("Eng ÷ Likes Ratio", f"{safe_float(eqr_post):.3f}",
                    "engagement per like (interaction depth)",   "#fbbf24"), unsafe_allow_html=True)

    # ── PART A: Overall charts ───────────────────────────────
    sec("Post Volume & Content Distribution")
    ch1, ch2 = st.columns([3,2])

    with ch1:
        me = posts.groupby("Month")["Engagement"].sum().reset_index().sort_values("Month")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=me["Month"], y=me["Engagement"],
            marker_color="#4f8ef7", marker_line_width=0, opacity=0.85, name="Engagement"))
        fig.add_trace(go.Scatter(x=me["Month"], y=me["Engagement"],
            mode="lines", line=dict(color="#a3c4fd",width=1.5), showlegend=False))
        dark(fig, 300)
        fig.update_layout(title=dict(text="Monthly Post Engagement",
                                     font=dict(size=12,color="#5c6488")), bargap=0.22)
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # ── NEW: Likes vs Comments — Quadrant Bubble Chart ──
        # Instead of a plain scatter, bucket posts into 4 engagement
        # quadrants and show bubble size = total engagement
        if "Likes" in posts.columns and "Comments" in posts.columns:
            sp = posts.sample(min(1200, len(posts)), random_state=7).copy()
            med_likes    = sp["Likes"].median()
            med_comments = sp["Comments"].median()

            def _quadrant(row):
                hi_l = row["Likes"]    >= med_likes
                hi_c = row["Comments"] >= med_comments
                if hi_l and hi_c:   return "High Likes & Comments"
                if hi_l:            return "High Likes, Low Comments"
                if hi_c:            return "Low Likes, High Comments"
                return "Low Likes & Comments"

            sp["Quadrant"] = sp.apply(_quadrant, axis=1)
            quad_colors = {
                "High Likes & Comments":      "#34d399",
                "High Likes, Low Comments":   "#4f8ef7",
                "Low Likes, High Comments":   "#fbbf24",
                "Low Likes & Comments":       "#f87171",
            }
            fig2 = px.scatter(
                sp, x="Likes", y="Comments",
                color="Quadrant",
                size="Engagement",
                size_max=18,
                color_discrete_map=quad_colors,
                opacity=0.65,
                hover_data=(["Handle","Post_Date","Engagement"]
                            if "Handle" in sp.columns else ["Post_Date","Engagement"]),
            )
            # quadrant dividers
            fig2.add_vline(x=med_likes,    line_dash="dot", line_color="#2a2f45", line_width=1.2)
            fig2.add_hline(y=med_comments, line_dash="dot", line_color="#2a2f45", line_width=1.2)
            # quadrant labels
            x_max = sp["Likes"].quantile(0.97)
            y_max = sp["Comments"].quantile(0.97)
            for txt, x, y, col in [
                ("🔥 Top",         x_max*0.82, y_max*0.90, "#34d399"),
                ("👍 Likes",       x_max*0.82, y_max*0.12, "#4f8ef7"),
                ("💬 Comments",    x_max*0.08, y_max*0.90, "#fbbf24"),
                ("📉 Low",         x_max*0.08, y_max*0.12, "#f87171"),
            ]:
                fig2.add_annotation(x=x, y=y, text=txt,
                    showarrow=False, font=dict(color=col, size=10),
                    bgcolor="rgba(13,15,20,0.6)", borderpad=3)
            fig2.update_traces(marker_line_width=0)
            dark(fig2, 300)
            fig2.update_layout(
                title=dict(text="Likes vs Comments — Quadrant View (bubble = engagement)",
                           font=dict(size=12,color="#5c6488")),
                legend=dict(orientation="v", x=1.01, y=1,
                            font=dict(size=9), itemsizing="constant"))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Engagement Heatmap ───────────────────────────────────
    sec("Engagement Heatmap — Month × Category")
    if "Category_Name" in posts.columns:
        pivot = (posts.groupby(["MonthName","Category_Name"])["Engagement"]
                 .sum().unstack("Category_Name").fillna(0))
        fig3 = px.imshow(pivot,
                         color_continuous_scale=["#0d0f14","#1a3f8f","#4f8ef7","#a3c4fd"],
                         aspect="auto")
        dark(fig3, 280)
        fig3.update_layout(title=dict(text="Total Engagement by Month & Category",
                                      font=dict(size=12,color="#5c6488")),
                           coloraxis_colorbar=dict(tickfont=dict(color="#9ba3c4")))
        st.plotly_chart(fig3, use_container_width=True)

    sec("☁️ Trending Hashtags")
    if "Hashtags" in posts.columns:
        text = " ".join(posts["Hashtags"].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=1200, height=380, background_color="#141720",
                           colormap="cool", prefer_horizontal=0.75, max_words=70).generate(text)
            fig_wc, ax = plt.subplots(figsize=(14,4))
            fig_wc.patch.set_facecolor("#141720"); ax.set_facecolor("#141720")
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
        else:
            st.info("No hashtag data in current filter.")

        # ── Full post table for this handle ────────────────
        sec(f"All Posts by @{handle}  ({len(handle_posts)} total)")
        show_cols = [c for c in ["Post_ID","Post_Date","Likes","Comments",
                                  "Engagement","Hashtags"] if c in handle_posts.columns]
        disp = handle_posts.sort_values("Post_Date", ascending=False)[show_cols].copy()
        if "Post_Date" in disp.columns:
            disp["Post_Date"] = disp["Post_Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=300)    

# ==========================================================
# PAGE 4 — AI LEAD SCORING
# ==========================================================
elif page == "AI Lead Scoring":
    hq  = (leads["Lead_Quality"]=="high").sum()
    med = (leads["Lead_Quality"]=="medium").sum()
    low = (leads["Lead_Quality"]=="low").sum()
    tot = max(len(leads),1)

    st.markdown(
        f'<p style="color:#5c6488;font-size:13px;margin-bottom:1rem">'
        f'AI-driven outreach prioritisation — score, segment, and act · '
        f'<b style="color:#4f8ef7">{len(leads):,} records in view</b></p>',
        unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("🔥 High Priority",  fmt(hq),  f"{hq/tot*100:.1f}% · Contact now","#34d399"), unsafe_allow_html=True)
    k2.markdown(kpi("⏳ Medium Priority", fmt(med), f"{med/tot*100:.1f}% · Nurture",   "#fbbf24"), unsafe_allow_html=True)
    k3.markdown(kpi("🗑 Low Priority",    fmt(low), f"{low/tot*100:.1f}% · Ignore",    "#f87171"), unsafe_allow_html=True)
    k4.markdown(kpi("Avg Lead Score",  f"{leads['Lead_Score'].mean():.1f}",
                    "out of 100 · current view", "#4f8ef7"), unsafe_allow_html=True)

    with st.expander("🧮 How is Lead Score calculated?", expanded=False):
        st.markdown("""
```
Step 1 — Raw Score (0 to ~68)
  Engagement_Rate  × 0.40   →  rewards active audiences
  Follower_Count/2M × 30    →  capped at 2M followers
  Avg Sentiment norm × 20   →  from post_metrics.csv  (-1 to +1 → 0 to 20)
  SaaS Relevance   × 10     →  from category_dim.csv

Step 2 — Normalise to 0–100
  Lead_Score = (raw − min) / (max − min) × 100

Step 3 — Quality Tier (fixed thresholds on 0–100 scale)
  High   : Score > 60   (~21% of influencers)
  Medium : 30 < Score ≤ 60   (~52% of influencers)
  Low    : Score ≤ 30   (~28% of influencers)
```""")

    sec("Pipeline Funnel & Score Distribution")
    ch1, ch2 = st.columns([2, 3])

    with ch1:
        funnel = pd.DataFrame({
            "Stage":["All Leads","Med + High","High Only"],
            "Count":[tot, med+hq, hq]})
        fig = go.Figure(go.Funnel(
            y=funnel["Stage"], x=funnel["Count"],
            marker_color=["#1d4ed8","#fbbf24","#34d399"],
            textinfo="value+percent initial",
            connector=dict(fillcolor="#0d0f14")))
        dark(fig, 320)
        fig.update_layout(title=dict(text="Lead Pipeline Funnel",
                                      font=dict(size=12,color="#5c6488")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig_sns, ax = plt.subplots(figsize=(8, 4.5))
        fig_sns.patch.set_facecolor("#141720")
        ax.set_facecolor("#141720")
        palette = {"high": "#34d399", "medium": "#fbbf24", "low": "#f87171"}
        for quality, color in palette.items():
            subset = leads[leads["Lead_Quality"] == quality]["Lead_Score"].dropna()
            if len(subset) > 0:
                sns.histplot(subset, ax=ax, bins=30, color=color, alpha=0.55,
                             label=quality.title(), edgecolor="none", kde=True,
                             line_kws=dict(linewidth=2, color=color))
        ax.axvline(30, color="#5c6488", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(60, color="#5c6488", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(15, ax.get_ylim()[1]*0.92, "Low",    color="#f87171", fontsize=9, ha="center", fontfamily="monospace")
        ax.text(45, ax.get_ylim()[1]*0.92, "Medium", color="#fbbf24", fontsize=9, ha="center", fontfamily="monospace")
        ax.text(75, ax.get_ylim()[1]*0.92, "High",   color="#34d399", fontsize=9, ha="center", fontfamily="monospace")
        ax.set_xlabel("Lead Score", color="#9ba3c4", fontsize=11)
        ax.set_ylabel("Count",      color="#9ba3c4", fontsize=11)
        ax.tick_params(colors="#5c6488", labelsize=10)
        for spine in ax.spines.values(): spine.set_edgecolor("#1c2030")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#1c2030", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, labelcolor="#9ba3c4", fontsize=10)
        ax.set_title("Lead Score Distribution by Quality Tier",
                     color="#5c6488", fontsize=11, pad=12, loc="left")
        plt.tight_layout()
        st.pyplot(fig_sns, use_container_width=True)
        plt.close(fig_sns)

    sec("Score Distribution by Category")
    if "Category_Name" in leads.columns:
        fig_v = go.Figure()
        for cat, color in CAT_CLR.items():
            subset = leads[leads["Category_Name"]==cat]["Lead_Score"].dropna()
            if len(subset) > 0:
                fig_v.add_trace(go.Violin(
                    x=[cat]*len(subset), y=subset,
                    name=cat, line_color=color,
                    fillcolor=color, opacity=0.35,
                    box_visible=True, meanline_visible=True,
                    points=False))
        dark(fig_v, 300)
        fig_v.update_layout(
            title=dict(text="Lead Score Violin — spread and median per category",
                       font=dict(size=12,color="#5c6488")),
            showlegend=False, violingap=0.25)
        st.plotly_chart(fig_v, use_container_width=True)

    sec("🎯 Priority Outreach List")
    f1, f2, f3 = st.columns(3)
    cat_opts = ["All"] + (sorted(leads["Category_Name"].dropna().unique().tolist())
                          if "Category_Name" in leads.columns else [])
    fcat = f1.selectbox("Category", cat_opts, key="sc_cat")
    fact = f2.selectbox("Action", ["All","🔥 Contact Now","⏳ Nurture","🗑 Ignore"], key="sc_act")
    fn   = f3.selectbox("Show", [20,50,100], key="sc_n")

    pt = leads.copy()
    if fcat!="All" and "Category_Name" in pt.columns: pt = pt[pt["Category_Name"]==fcat]
    if fact!="All" and "Action"        in pt.columns: pt = pt[pt["Action"]==fact]

    show_c = [c for c in ["Handle","Category_Name","Lead_Score","Lead_Quality",
                           "Action","Follower_Count","Engagement_Rate","Avg_Sentiment"]
              if c in pt.columns]
    pt = pt.nlargest(fn,"Lead_Score")[show_c].rename(columns={
        "Category_Name":"Category","Lead_Score":"Score","Lead_Quality":"Quality",
        "Follower_Count":"Followers","Engagement_Rate":"ER %","Avg_Sentiment":"Avg Sentiment"})
    if "Followers"     in pt.columns: pt["Followers"]     = pt["Followers"].apply(fmt)
    if "Score"         in pt.columns: pt["Score"]         = pt["Score"].round(1)
    if "ER %"          in pt.columns: pt["ER %"]          = pt["ER %"].round(2)
    if "Avg Sentiment" in pt.columns: pt["Avg Sentiment"] = pt["Avg Sentiment"].round(3)

    st.dataframe(pt.reset_index(drop=True), use_container_width=True, height=380)
    st.download_button("⬇️ Export as CSV",
                       pt.to_csv(index=False).encode(),
                       "priority_leads.csv","text/csv")


# ==========================================================
# PAGE 5 — ABOUT
# ==========================================================
elif page == "About":
    total_inf   = len(leads_full)
    total_posts = len(posts_full)
    cats_count  = leads_full["Category_Name"].nunique() if "Category_Name" in leads_full.columns else 5

    st.markdown(f"""
    <div class="about-hero">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <div style="font-size:1.6rem;font-weight:700;color:#e8eaf6;margin-bottom:6px;">⚡ InfluenceIQ</div>
          <div style="font-size:13px;color:#9ba3c4;max-width:640px;line-height:1.7;">
            InfluenceIQ is an advanced <b style="color:#e8eaf6">AI-powered influencer intelligence platform</b>
            built with Streamlit and Python. It helps founders, agencies, and marketing teams detect
            high-quality influencer leads, analyse engagement authenticity, track individual posts, and
            prioritise outreach — all powered by real CSV data.
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;">
          <div style="background:#1c2030;border:1px solid #2a2f45;border-radius:10px;padding:12px 18px;text-align:center;">
            <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:#4f8ef7">{total_inf:,}</div>
            <div style="font-size:10px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Influencers</div>
          </div>
          <div style="background:#1c2030;border:1px solid #2a2f45;border-radius:10px;padding:12px 18px;text-align:center;">
            <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:#34d399">{total_posts:,}</div>
            <div style="font-size:10px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Posts</div>
          </div>
          <div style="background:#1c2030;border:1px solid #2a2f45;border-radius:10px;padding:12px 18px;text-align:center;">
            <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:#fbbf24">{cats_count}</div>
            <div style="font-size:10px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Categories</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="about-card about-card-blue">
          <div class="about-title">📊 Dashboard Sections</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#4f8ef7"></div>
            <span><b style="color:#e8eaf6">Executive Overview</b> — KPI cards, smart insights, engagement trend, quality donut, engagement quality ratio chart</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span><b style="color:#e8eaf6">Lead Intelligence</b> — Followers vs ER scatter, FF ratio bars, tier stacked bar, ER box plot, ranked table</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#a78bfa"></div>
            <span><b style="color:#e8eaf6">Post Analytics</b> — Monthly engagement, quadrant bubble chart, heatmap, hashtag cloud + Post Inspector with engagement history</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#f87171"></div>
            <span><b style="color:#e8eaf6">AI Lead Scoring</b> — Pipeline funnel, seaborn histogram, violin chart, priority outreach export</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card about-card-green">
          <div class="about-title">🔑 Key Features</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span>Multi-filter sidebar — category, quality, ER range, followers, tier, date, year</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span>Empty multiselect = <b style="color:#e8eaf6">show all data</b>. No accidental zero-state.</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span><b style="color:#e8eaf6">Post Inspector</b> inside Post Analytics — search any Post ID or Handle, see history with selected post ★ highlighted</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span>Likes vs Comments as a <b style="color:#e8eaf6">quadrant bubble chart</b> — colour-coded by engagement zone</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span>Lead scores <b style="color:#e8eaf6">normalised 0–100</b> using min-max scaling</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#34d399"></div>
            <span>Priority outreach list with one-click <b style="color:#e8eaf6">CSV export</b></span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="about-card about-card-amber">
          <div class="about-title">🗂 Data Model</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div>
            <span><span class="about-tag">influencer_master.csv</span> — {total_inf:,} records · core profiles</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div>
            <span><span class="about-tag">post_metrics.csv</span> — {total_posts:,} records · likes, comments, hashtags</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div>
            <span><span class="about-tag">category_dim.csv</span> — {cats_count} categories · SaaS relevance weights</span></div>
          <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1c2030">
            <div style="font-size:10px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Join Keys</div>
            <div style="font-size:12px;color:#9ba3c4;line-height:1.8;">
              <span class="about-tag">influencer_master</span> → <span class="about-tag">category_dim</span> via <b style="color:#e8eaf6">Category_ID</b><br>
              <span class="about-tag">post_metrics</span> → <span class="about-tag">influencer_master</span> via <b style="color:#e8eaf6">Handle</b>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="about-card about-card-pink">
          <div class="about-title">🤖 Lead Score Formula</div>
          <hr class="about-divider">
          <div style="font-family:'DM Mono',monospace;font-size:11.5px;color:#9ba3c4;
            background:#0d0f14;border:1px solid #1c2030;border-radius:8px;
            padding:14px 16px;line-height:2;">
            <span style="color:#5c6488"># Step 1 — Weighted raw score</span><br>
            ER_score   = Engagement_Rate × <span style="color:#4f8ef7">0.40</span><br>
            Fol_score  = clip(Followers, 2M) / 2M × <span style="color:#34d399">30</span><br>
            Sent_score = norm(Avg_Sentiment) × <span style="color:#a78bfa">20</span><br>
            SaaS_score = SaaS_Relevance × <span style="color:#fbbf24">10</span><br><br>
            <span style="color:#5c6488"># Step 2 — Normalise 0 → 100</span><br>
            Score = (raw − min) / (max − min) × <span style="color:#f472b6">100</span><br><br>
            <span style="color:#5c6488"># Step 3 — Quality tiers</span><br>
            <span style="color:#34d399">High</span> ≥ 60 &nbsp;·&nbsp;
            <span style="color:#fbbf24">Medium</span> 30–59 &nbsp;·&nbsp;
            <span style="color:#f87171">Low</span> &lt; 30
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card" style="border-left:4px solid #34d399;">
      <div class="about-title">🚀 How to Run</div>
      <hr class="about-divider">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
        <div>
          <div style="font-size:11px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Folder Structure</div>
          <div style="font-family:'DM Mono',monospace;font-size:11.5px;color:#9ba3c4;
            background:#0d0f14;border:1px solid #1c2030;border-radius:8px;padding:14px 16px;line-height:2;">
            <span style="color:#4f8ef7">Project/</span><br>
            &nbsp;&nbsp;<span style="color:#e8eaf6">project.py</span><br>
            &nbsp;&nbsp;<span style="color:#4f8ef7">data/</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#34d399">influencer_master.csv</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#34d399">post_metrics.csv</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#34d399">category_dim.csv</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#34d399">lead_scoring.csv</span>
          </div>
        </div>
        <div>
          <div style="font-size:11px;color:#5c6488;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Run Command</div>
          <div style="font-family:'DM Mono',monospace;font-size:11.5px;color:#9ba3c4;
            background:#0d0f14;border:1px solid #1c2030;border-radius:8px;padding:14px 16px;line-height:2;">
            pip install seaborn<br>
            <span style="color:#34d399">streamlit run project.py</span><br><br>
            <span style="color:#5c6488"># opens at</span><br>
            <span style="color:#4f8ef7">http://localhost:8501</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#2a2f45;'
    'font-family:\'DM Mono\',monospace">'
    'InfluenceIQ · AI Influencer Intelligence · Streamlit + Plotly + Seaborn</div>',
    unsafe_allow_html=True)