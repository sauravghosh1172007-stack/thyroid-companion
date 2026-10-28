from __future__ import annotations

APP_DESCRIPTION = """Thyroid Companion — polished Streamlit interface for an educational ML prototype.

The model was trained on synthetic, computer-generated labels. This application is
not a medical diagnostic tool and must not be used to diagnose or rule out disease.
"""

from pathlib import Path
from datetime import datetime
import csv
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "thyroid_symptom_model.joblib"
RESULTS_DIR = BASE / "results"
METRICS_PATH = RESULTS_DIR / "binary_metrics.json"
IMPORTANCE_PATH = RESULTS_DIR / "binary_feature_importance.csv"
CM_IMAGE_PATH = RESULTS_DIR / "binary_confusion_matrix.png"
IMPORTANCE_IMAGE_PATH = RESULTS_DIR / "binary_feature_importance.png"
FEEDBACK_PATH = RESULTS_DIR / "ux_feedback.csv"

st.set_page_config(
    page_title="Thyroid Companion | AI Applications Lab",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS is intentional here: the requested deliverable is a branded, polished UI.
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"] {
        background: #F4F7F6 !important;
        color: #17312F !important;
    }
    [data-testid="stHeader"] { background: rgba(244,247,246,.94) !important; }
    .block-container { max-width: 1320px; padding-top: 1.25rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"] {
        color: #183431;
    }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p { color: #536A64 !important; }
    .stApp [data-testid="stMarkdownContainer"] p { color: #183431; }
    .stApp [data-testid="stMarkdownContainer"] li { color: #284944; }
    .stApp [data-testid="stMarkdownContainer"] a { color: #126A5E !important; text-decoration-thickness: 1px; }
    .stApp small { color: #536A64 !important; }
    .stApp hr { border-color: #DCE6E3; }

    .tc-hero {
        position: relative; overflow: hidden;
        background: linear-gradient(118deg, #123D39 0%, #17675D 56%, #477A8B 100%);
        color: #FFFFFF; border-radius: 26px; padding: 2rem 2.15rem;
        margin: .15rem 0 1.1rem 0;
        box-shadow: 0 16px 42px rgba(16, 62, 56, .14);
    }
    .tc-hero:after {
        content: ""; position: absolute; width: 300px; height: 300px; right: -95px; top: -155px;
        border: 1px solid rgba(255,255,255,.16); border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
        pointer-events: none;
    }
    .tc-brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .8rem; }
    .tc-mark { display:flex; align-items:center; justify-content:center; width:42px; height:42px;
        border-radius:14px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.2);
        font-size:1.35rem; }
    .tc-brand-name { color:#FFFFFF !important; font-weight:780; font-size:1.72rem; letter-spacing:-.045em; line-height:1.12; }
    .tc-kicker { color:#BDE7DB !important; text-transform:uppercase; font-size:.72rem; letter-spacing:.13em; font-weight:750; }
    .tc-hero-copy { color:#E3F3EF !important; font-size:1rem; line-height:1.6; max-width:760px; margin:.65rem 0 0 0; }
    .tc-hero-meta { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }
    .tc-pill { display:inline-flex; align-items:center; gap:.35rem; border-radius:999px; padding:.38rem .7rem;
        font-size:.75rem; font-weight:650; color:#F0FBF8; background:rgba(255,255,255,.10);
        border:1px solid rgba(255,255,255,.18); }

    .tc-notice { display:flex; gap:.8rem; align-items:flex-start; background:#FFF8DD;
        color:#604B11; border:1px solid #F1E1A6; border-radius:15px; padding:.85rem 1rem;
        margin:.85rem 0 1.2rem 0; font-size:.88rem; line-height:1.55; }
    .tc-notice strong { color:#543E08; }
    .tc-notice-icon { font-size:1.1rem; line-height:1.5; }
    .tc-section-title { display:flex; align-items:flex-start; gap:.75rem; margin:.25rem 0 .9rem 0; }
    .tc-step { display:flex; align-items:center; justify-content:center; min-width:31px; height:31px;
        border-radius:10px; background:#E2F3EE; color:#146356; font-size:.78rem; font-weight:800; }
    .tc-section-title h3 { font-size:1.06rem; margin:0; color:#193B36 !important; letter-spacing:-.02em; }
    .tc-section-title p { font-size:.84rem; color:#536A64 !important; margin:.2rem 0 0 0; }
    .tc-panel-title { display:flex; align-items:center; gap:.55rem; margin:0 0 .25rem 0; }
    .tc-panel-title h3 { margin:0; font-size:1.02rem; letter-spacing:-.02em; }
    .tc-muted { color:#536A64 !important; font-size:.84rem; line-height:1.55; }
    .tc-result-positive { border:1px solid #E9D8A3; background:#FFFBEB; border-radius:18px; padding:1.2rem 1.3rem; }
    .tc-result-negative { border:1px solid #D6E8E2; background:#F1FAF6; border-radius:18px; padding:1.2rem 1.3rem; }
    .tc-result-kicker { text-transform:uppercase; font-size:.71rem; font-weight:800; letter-spacing:.11em; color:#526862 !important; }
    .tc-result-positive h3, .tc-result-negative h3 { margin:.3rem 0 .45rem 0; font-size:1.22rem; color:#183431 !important; }
    .tc-result-positive p, .tc-result-negative p { margin:0; font-size:.9rem; line-height:1.6; color:#465C57 !important; }
    .tc-tag { display:inline-block; border-radius:8px; padding:.3rem .55rem; font-size:.72rem; font-weight:750; background:#E7F3EF; color:#14564D !important; }
    .tc-footer { text-align:center; border-top:1px solid #DCE6E3; padding-top:1.15rem; margin-top:2.2rem; color:#536A64 !important; font-size:.78rem; }
    .tc-flow { display:flex; flex-wrap:wrap; align-items:center; gap:.55rem; margin:.7rem 0 1.1rem 0; }
    .tc-flow span { background:#FFFFFF; border:1px solid #DCE7E3; border-radius:10px; padding:.55rem .7rem; color:#284944; font-size:.82rem; font-weight:650; }
    .tc-flow b { color:#7A918A; font-size:.9rem; }

    div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #DFE8E5; border-radius:16px;
        padding:.95rem 1rem; box-shadow:0 3px 12px rgba(25,60,53,.025); min-height:108px; }
    div[data-testid="stMetricLabel"] p { color:#657873 !important; font-size:.78rem !important; font-weight:600 !important; }
    div[data-testid="stMetricValue"] { color:#163D37 !important; font-size:1.6rem !important; font-weight:760 !important; letter-spacing:-.04em; }
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius:11px 11px 0 0; padding:.8rem 1rem; font-weight:650;
        color:#38564F !important; opacity:1 !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span { color:#38564F !important; opacity:1 !important; }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span { color:#126A5E !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#1B8C78 !important; }
    div[data-testid="stFormSubmitButton"] button, div.stButton > button, div[data-testid="stDownloadButton"] button {
        border-radius:11px; font-weight:700; min-height:2.75rem; transition:all .15s ease;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"], div.stButton > button[kind="primary"] {
        box-shadow:0 7px 18px rgba(20,118,101,.16);
    }
    .stApp input, .stApp textarea { background:#FFFFFF !important; color:#183431 !important; border-color:#CBD9D5 !important; }
    .stApp [data-baseweb="select"] > div { background:#FFFFFF !important; border-color:#CBD9D5 !important; }
    .stApp [data-baseweb="select"] div, .stApp [data-baseweb="select"] span { color:#183431 !important; }
    [data-baseweb="popover"], [role="listbox"], [role="option"] { background:#FFFFFF !important; color:#183431 !important; }
    [role="option"] * { color:#183431 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#DFE8E5 !important; border-radius:17px !important; }
    @media (max-width: 720px) {
        .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
        .tc-hero { padding:1.35rem; border-radius:20px; }
        .tc-brand-name { font-size:1.42rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data/model loading ────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Keep `thyroid_symptom_model.joblib` in the same "
        "folder as `app.py`, and keep the `results` folder beside them."
    )
    st.stop()


@st.cache_resource
def load_bundle(path: str, modified_ns: int):
    # mtime in the cache key lets a newly trained model reload after a restart/change.
    return joblib.load(path)


@st.cache_data
def load_json(path: str, modified_ns: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


@st.cache_data
def load_importance_csv(path: str, modified_ns: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"feature", "importance"}.issubset(df.columns):
        return pd.DataFrame(columns=["feature", "importance"])
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    return df.dropna(subset=["importance"]).sort_values("importance", ascending=False)


model_mtime = MODEL_PATH.stat().st_mtime_ns
bundle = load_bundle(str(MODEL_PATH), model_mtime)
metrics = load_json(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns) if METRICS_PATH.exists() else {}
importance_df = (
    load_importance_csv(str(IMPORTANCE_PATH), IMPORTANCE_PATH.stat().st_mtime_ns)
    if IMPORTANCE_PATH.exists()
    else pd.DataFrame(columns=["feature", "importance"])
)


# ── Question labels ───────────────────────────────────────────────────────────
FAMILY_QUESTIONS = {
    "Family_History": "Has anyone in your close family had a thyroid condition?",
    "Mother_Thyroid": "Has your mother had a thyroid condition?",
    "Father_Thyroid": "Has your father had a thyroid condition?",
    "Sibling_Thyroid": "Has a brother or sister had a thyroid condition?",
    "Autoimmune_Family": "Has a close family member had an autoimmune condition (for example, rheumatoid arthritis or type 1 diabetes)?",
}
SYMPTOM_QUESTIONS = {
    "Fatigue": "Have you felt unusually tired, even after resting?",
    "Weight_Gain": "Have you gained weight without an obvious reason?",
    "Weight_Loss": "Have you lost weight without trying?",
    "Cold_Intolerance": "Do you often feel colder than people around you?",
    "Heat_Intolerance": "Do you often feel unusually hot or sweat more than expected?",
    "Hair_Loss": "Have you noticed more hair shedding or thinning than usual?",
    "Dry_Skin": "Has your skin been unusually dry?",
    "Constipation": "Have you been dealing with constipation more than usual?",
    "Palpitations": "Have you noticed a racing, pounding or fluttering heartbeat?",
    "Tremor": "Have you noticed shaky hands or trembling?",
    "Anxiety": "Have you felt unusually nervous, anxious or on edge?",
    "Depression": "Have you had a persistent low mood?",
    "Brain_Fog": "Have you had trouble concentrating or felt mentally foggy?",
    "Neck_Swelling": "Have you noticed swelling or a lump at the front of your neck?",
    "Menstrual_Irregularity": "If applicable, have your periods become irregular?",
}
CHOICES = ["Not sure", "No", "Yes"]
FEATURE_LABELS = {
    "Age": "Age",
    "Family_History": "Family thyroid history",
    "Mother_Thyroid": "Mother thyroid history",
    "Father_Thyroid": "Father thyroid history",
    "Sibling_Thyroid": "Sibling thyroid history",
    "Autoimmune_Family": "Family autoimmune history",
    "Fatigue": "Fatigue",
    "Weight_Gain": "Weight gain",
    "Weight_Loss": "Weight loss",
    "Cold_Intolerance": "Cold intolerance",
    "Heat_Intolerance": "Heat intolerance",
    "Hair_Loss": "Hair loss",
    "Dry_Skin": "Dry skin",
    "Constipation": "Constipation",
    "Palpitations": "Palpitations",
    "Tremor": "Tremor",
    "Anxiety": "Anxiety",
    "Depression": "Low mood",
    "Brain_Fog": "Brain fog",
    "Neck_Swelling": "Neck swelling",
    "Menstrual_Irregularity": "Menstrual irregularity",
    "Gender_Female": "Gender: female",
    "Gender_Male": "Gender: male",
    "Gender_Not stated": "Gender: not stated",
}


def encode_answer(answer: str) -> float:
    if answer == "Yes":
        return 1.0
    if answer == "No":
        return 0.0
    return np.nan


def format_pct(value, digits: int = 1) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}%}"
    except (TypeError, ValueError):
        return "—"


def format_count(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def plotly_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=16, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#3C5550", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#173D37", font_color="#FFFFFF", bordercolor="#173D37"),
    )
    fig.update_xaxes(showline=False, zeroline=False, gridcolor="#E6EEEB", linecolor="#E6EEEB")
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#E6EEEB", linecolor="#E6EEEB")
    return fig


def render_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True}, key=key)


def friendly_feature_name(name: str) -> str:
    text = str(name)
    if text in FEATURE_LABELS:
        return FEATURE_LABELS[text]
    if text.startswith("Gender_"):
        return "Gender: " + text.removeprefix("Gender_")
    return text.replace("_", " ")


def reset_check() -> None:
    st.session_state.pop("last_result", None)
    st.session_state.pop("feedback_sent", None)
    for key in ["age_text", "gender_ui"]:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("fam_") or key.startswith("sym_"):
            st.session_state.pop(key, None)


def save_feedback(rating: int, comment: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    needs_header = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["timestamp", "usability_rating_1_to_5", "interface_feedback"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), rating, comment.strip()])


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <section class="tc-hero">
      <div class="tc-brand">
        <div class="tc-mark">🌿</div>
        <div>
          <div class="tc-kicker">AI applications lab · health intelligence</div>
          <div class="tc-brand-name">Thyroid Companion</div>
        </div>
      </div>
      <p class="tc-hero-copy">A calmer way to explore what you have noticed. Answer in everyday language—no blood-test numbers required. The model compares your answers with patterns in its synthetic training data.</p>
      <div class="tc-hero-meta">
        <span class="tc-pill">● Model ready</span>
        <span class="tc-pill">LightGBM classification</span>
        <span class="tc-pill">Symptom-first experience</span>
      </div>
    </section>
    <div class="tc-notice"><span class="tc-notice-icon">ⓘ</span><div><strong>Educational prototype only.</strong> This model was trained on synthetic, computer-generated records—not real patients. It cannot diagnose or rule out thyroid disease, and its output is not medical advice or a personal risk estimate.</div></div>
    """,
    unsafe_allow_html=True,
)

tab_check, tab_dashboard, tab_about = st.tabs(
    ["🌱  Symptom check", "📊  Model dashboard", "ⓘ  Project & methodology"]
)


# ── Symptom check ─────────────────────────────────────────────────────────────
with tab_check:
    st.markdown(
        '<div class="tc-section-title"><span class="tc-step">01</span><div><h3>Let’s start with what you know</h3><p>There are no right or wrong answers. “Not sure” is always an option.</p></div></div>',
        unsafe_allow_html=True,
    )

    with st.form("symptom_check_form", clear_on_submit=False):
        with st.container(border=True):
            st.markdown(
                '<div class="tc-section-title"><span class="tc-step">A</span><div><h3>A little about you</h3><p>Both fields are optional.</p></div></div>',
                unsafe_allow_html=True,
            )
            age_col, gender_col = st.columns(2)
            with age_col:
                age_text = st.text_input("Age (optional)", placeholder="e.g. 24", key="age_text")
            with gender_col:
                gender_ui = st.selectbox(
                    "Gender (optional)",
                    ["Prefer not to say", "Female", "Male"],
                    index=0,
                    key="gender_ui",
                )

        with st.expander("Family history · optional", expanded=False):
            st.caption("Family history can be helpful, but it is completely fine not to know.")
            family_answers = {}
            family_cols = st.columns(2)
            for index, (feature, question) in enumerate(FAMILY_QUESTIONS.items()):
                with family_cols[index % 2]:
                    family_answers[feature] = st.selectbox(
                        question, CHOICES, index=0, key=f"fam_{feature}"
                    )

        with st.container(border=True):
            st.markdown(
                '<div class="tc-section-title"><span class="tc-step">B</span><div><h3>How have you been feeling?</h3><p>Think about what you have personally noticed, rather than guessing what an AI expects.</p></div></div>',
                unsafe_allow_html=True,
            )
            symptom_answers = {}
            symptom_cols = st.columns(2)
            for index, (feature, question) in enumerate(SYMPTOM_QUESTIONS.items()):
                with symptom_cols[index % 2]:
                    symptom_answers[feature] = st.selectbox(
                        question, CHOICES, index=0, key=f"sym_{feature}"
                    )

        submitted = st.form_submit_button(
            "Generate my summary  →", type="primary", width="stretch"
        )

    reset_col, tip_col = st.columns([1, 3])
    with reset_col:
        st.button("Start a new check", on_click=reset_check, width="stretch")
    with tip_col:
        st.caption("Tip: answer based on your recent experience. You can leave any question as ‘Not sure’.")

    if submitted:
        st.session_state.pop("last_result", None)
        age = np.nan
        age_error = None
        try:
            age = float(age_text.strip()) if age_text.strip() else np.nan
            if not np.isnan(age) and not (1 <= age <= 120):
                age_error = "Please enter an age from 1 to 120, or leave it blank."
        except ValueError:
            age_error = "Please enter your age as a number, or leave it blank."

        if age_error:
            st.error(age_error)
        else:
            known_symptoms = sum(value != "Not sure" for value in symptom_answers.values())
            known_family = sum(value != "Not sure" for value in family_answers.values())
            if known_symptoms < 3:
                st.error("Please answer at least three symptom questions. ‘Not sure’ is fine for everything else.")
            else:
                gender = gender_ui if gender_ui in ("Female", "Male") else "Not stated"
                input_row = {
                    "Age": age,
                    "Gender": gender,
                    **{feature: encode_answer(value) for feature, value in family_answers.items()},
                    **{feature: encode_answer(value) for feature, value in symptom_answers.items()},
                }
                try:
                    input_df = pd.DataFrame([input_row]).reindex(columns=bundle["features"])
                    transformed = bundle["preprocessor"].transform(input_df)
                    transformed = pd.DataFrame(
                        transformed,
                        columns=bundle["preprocessor"].get_feature_names_out(),
                    )
                    score = float(bundle["model"].predict_proba(transformed)[0, 1])
                    threshold = float(
                        bundle.get(
                            "positive_threshold",
                            metrics.get("threshold_selected_on_validation_for_accuracy", 0.5),
                        )
                    )
                    result = {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "age": age_text.strip() or "Not shared",
                        "gender": gender_ui,
                        "input_row": input_row,
                        "score": score,
                        "threshold": threshold,
                        "is_positive": score >= threshold,
                        "yes_symptoms": [
                            question for feature, question in SYMPTOM_QUESTIONS.items()
                            if symptom_answers[feature] == "Yes"
                        ],
                        "yes_family": [
                            question for feature, question in FAMILY_QUESTIONS.items()
                            if family_answers[feature] == "Yes"
                        ],
                        "symptom_answers": symptom_answers,
                        "family_answers": family_answers,
                        "known_symptoms": known_symptoms,
                        "known_family": known_family,
                    }
                    st.session_state["last_result"] = result
                except Exception as exc:
                    st.error(
                        "The model could not process these answers. Please check that the model, "
                        "training files and app are from the same project version."
                    )
                    with st.expander("Technical details for troubleshooting"):
                        st.code(f"{type(exc).__name__}: {exc}")

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.markdown("---")
        st.markdown(
            '<div class="tc-section-title"><span class="tc-step">02</span><div><h3>Your summary</h3><p>A plain-language recap of how the model classified this input pattern.</p></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            if res["is_positive"]:
                st.markdown(
                    '<div class="tc-result-positive"><div class="tc-result-kicker">Dataset pattern · non-Healthy / risk group</div><h3>Your answers overlap with a non-Healthy pattern in the dataset</h3><p>This is not a diagnosis. The positive label includes synthetic risk and phenotype categories, and the same symptoms can have many causes.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="tc-result-negative"><div class="tc-result-kicker">Dataset pattern · Healthy-like group</div><h3>Your answers are closer to the dataset’s Healthy-like pattern</h3><p>This does not rule out thyroid disease. The model may miss patterns and has not been clinically validated.</p></div>',
                    unsafe_allow_html=True,
                )
            st.caption("The model’s internal score is not displayed as a personal risk percentage because it is not calibrated to real-world disease probability.")
            st.write("")
            result_metric_cols = st.columns(3)
            result_metric_cols[0].metric("Symptoms answered", f"{res['known_symptoms']} / {len(SYMPTOM_QUESTIONS)}")
            result_metric_cols[1].metric("Symptoms marked yes", str(len(res["yes_symptoms"])))
            result_metric_cols[2].metric("Family-history answers known", f"{res['known_family']} / {len(FAMILY_QUESTIONS)}")

            yes_count = sum(answer == "Yes" for answer in res["symptom_answers"].values())
            no_count = sum(answer == "No" for answer in res["symptom_answers"].values())
            unsure_count = sum(answer == "Not sure" for answer in res["symptom_answers"].values())
            chart_col, list_col = st.columns([1, 1.15])
            with chart_col:
                st.markdown("#### Your response overview")
                st.caption("This chart summarizes your answers only; it is not a medical indicator.")
                response_fig = go.Figure(
                    go.Pie(
                        labels=["Yes", "No", "Not sure"],
                        values=[yes_count, no_count, unsure_count],
                        hole=0.72,
                        sort=False,
                        marker=dict(colors=["#168575", "#B9D8D0", "#E8B95E"], line=dict(color="#FFFFFF", width=3)),
                        textinfo="label+value",
                        hovertemplate="%{label}: %{value} questions<extra></extra>",
                    )
                )
                response_fig.add_annotation(
                    text=f"{len(SYMPTOM_QUESTIONS)}<br><span style='font-size:11px'>questions</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=19, color="#173D37")
                )
                plotly_layout(response_fig, 260)
                response_fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5))
                render_plot(response_fig, "personal_response_distribution")
            with list_col:
                st.markdown("#### What you marked yes")
                if res["yes_symptoms"]:
                    for item in res["yes_symptoms"]:
                        st.markdown(f"- {item}")
                else:
                    st.markdown('<p class="tc-muted">No symptom was marked “Yes”.</p>', unsafe_allow_html=True)
                if res["yes_family"]:
                    with st.expander("Family-history answers marked yes"):
                        for item in res["yes_family"]:
                            st.write(f"- {item}")

            st.info(
                "If symptoms persist, worsen or worry you, consider discussing them with a qualified healthcare professional. "
                "They can decide whether thyroid blood tests or other assessment is appropriate."
            )

            report_lines = [
                "THYROID COMPANION — EDUCATIONAL SYMPTOM SUMMARY",
                f"Created: {res['created_at']}",
                "",
                "Pattern output: " + (
                    "Answers overlap with a non-Healthy phenotype/risk pattern in the synthetic dataset."
                    if res["is_positive"]
                    else "Answers are closer to the synthetic dataset's Healthy-like pattern."
                ),
                "Age: " + str(res["age"]),
                "Gender: " + str(res["gender"]),
                "",
                "Symptom responses:",
            ]
            report_lines += [f"- {feature}: {answer}" for feature, answer in res["symptom_answers"].items()]
            report_lines += ["", "Family-history responses:"]
            report_lines += [f"- {feature}: {answer}" for feature, answer in res["family_answers"].items()]
            report_lines += [
                "", "IMPORTANT LIMITATIONS:",
                "- The model was trained on synthetic, computer-generated data, not real patient records.",
                "- This is not a diagnosis, medical advice, or a real-world estimate of disease risk.",
                "- A negative-pattern result does not rule out thyroid disease.",
                "- Seek qualified medical advice for persistent or concerning symptoms.",
            ]
            st.download_button(
                "Download my symptom summary",
                data="\n".join(report_lines),
                file_name="thyroid_companion_summary.txt",
                mime="text/plain",
                width="stretch",
            )


# ── Model dashboard ───────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown(
        '<div class="tc-section-title"><span class="tc-step">03</span><div><h3>Model performance at a glance</h3><p>Evaluation metrics and diagnostics for the current saved model.</p></div></div>',
        unsafe_allow_html=True,
    )
    if not metrics:
        st.info("No evaluation report was found. Keep the `results` folder beside `app.py`.")
    else:
        st.markdown(
            f'<span class="tc-tag">{metrics.get("model", "Binary classifier")}</span> '
            f'<span class="tc-tag">{format_count(metrics.get("source_rows_used", metrics.get("sample_rows_used", 0)))} source records</span> '
            f'<span class="tc-tag">{len(bundle.get("features", []))} user-answerable inputs</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Test accuracy", format_pct(metrics.get("accuracy")))
        kpi2.metric("Balanced accuracy", format_pct(metrics.get("balanced_accuracy")))
        kpi3.metric("Positive-pattern recall", format_pct(metrics.get("recall_positive")))
        kpi4.metric("Positive-pattern precision", format_pct(metrics.get("precision_positive")))
        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        kpi5.metric("Positive-pattern F1", format_pct(metrics.get("f1_positive")))
        kpi6.metric("Healthy-class specificity", format_pct(metrics.get("specificity_healthy")))
        kpi7.metric("ROC-AUC", f"{float(metrics['roc_auc']):.3f}" if metrics.get("roc_auc") is not None else "—")
        kpi8.metric("Evaluation records", format_count(metrics.get("test_rows")))

        st.info(
            "These scores measure agreement with computer-generated labels, not clinical accuracy. "
            "The same test split has been reviewed during several development experiments, so treat these as development results—not as a fresh independent final benchmark."
        )

        st.markdown("### Model diagnostics")
        cm_col, metric_col = st.columns([1.05, 1])
        with cm_col:
            with st.container(border=True):
                st.markdown('<div class="tc-panel-title"><h3>Confusion matrix</h3></div>', unsafe_allow_html=True)
                st.caption("Rows = actual synthetic label · columns = model output")
                cm = metrics.get("confusion_matrix")
                if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
                    cm_values = [[int(value) for value in row] for row in cm]
                    cm_text = [[format_count(value) for value in row] for row in cm_values]
                    cm_fig = go.Figure(
                        go.Heatmap(
                            z=cm_values,
                            x=["Predicted healthy-like", "Predicted non-Healthy"],
                            y=["Actual healthy-like", "Actual non-Healthy"],
                            text=cm_text,
                            texttemplate="%{text}",
                            colorscale=[[0, "#E9F4F0"], [0.5, "#8CC9BB"], [1, "#176C60"]],
                            showscale=False,
                            hovertemplate="%{y}<br>%{x}<br>Records: %{z:,}<extra></extra>",
                        )
                    )
                    plotly_layout(cm_fig, 320)
                    cm_fig.update_layout(margin=dict(l=5, r=5, t=5, b=50), xaxis_title="Model output", yaxis_title="True label")
                    cm_fig.update_yaxes(autorange="reversed", showgrid=False)
                    cm_fig.update_xaxes(showgrid=False, tickangle=-10)
                    render_plot(cm_fig, "model_confusion_matrix")
                elif CM_IMAGE_PATH.exists():
                    st.image(str(CM_IMAGE_PATH), width="stretch")
                else:
                    st.caption("Confusion matrix data is not available in the saved metrics.")

        with metric_col:
            with st.container(border=True):
                st.markdown('<div class="tc-panel-title"><h3>Metric profile</h3></div>', unsafe_allow_html=True)
                st.caption("Higher is better for these metrics; use recall and precision together.")
                metric_rows = [
                    ("Accuracy", metrics.get("accuracy")),
                    ("Balanced accuracy", metrics.get("balanced_accuracy")),
                    ("Positive precision", metrics.get("precision_positive")),
                    ("Positive recall", metrics.get("recall_positive")),
                    ("Positive F1", metrics.get("f1_positive")),
                    ("Healthy specificity", metrics.get("specificity_healthy")),
                ]
                metric_rows = [(name, float(value)) for name, value in metric_rows if value is not None]
                if metric_rows:
                    metric_rows = metric_rows[::-1]
                    profile_fig = go.Figure(
                        go.Bar(
                            x=[value * 100 for _, value in metric_rows],
                            y=[name for name, _ in metric_rows],
                            orientation="h",
                            marker=dict(color=["#78BBAA", "#87B9C5", "#1B8475", "#509C90", "#A0CFC3", "#3D7285"][:len(metric_rows)], line=dict(width=0)),
                            text=[f"{value:.1%}" for _, value in metric_rows],
                            textposition="outside",
                            cliponaxis=False,
                            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                        )
                    )
                    plotly_layout(profile_fig, 320)
                    profile_fig.update_layout(margin=dict(l=8, r=42, t=5, b=25), showlegend=False)
                    profile_fig.update_xaxes(range=[0, 108], ticksuffix="%", showgrid=True, gridcolor="#E7EFEC")
                    profile_fig.update_yaxes(showgrid=False)
                    render_plot(profile_fig, "model_metric_profile")

        feature_col, distribution_col = st.columns([1.2, .8])
        with feature_col:
            with st.container(border=True):
                st.markdown('<div class="tc-panel-title"><h3>Inputs used most by the model</h3></div>', unsafe_allow_html=True)
                st.caption("Tree split importance is descriptive; it does not prove cause or medical significance.")
                if not importance_df.empty:
                    top = importance_df.head(12).copy()
                    top["label"] = top["feature"].map(friendly_feature_name)
                    top = top.sort_values("importance", ascending=True)
                    importance_fig = go.Figure(
                        go.Bar(
                            x=top["importance"], y=top["label"], orientation="h",
                            marker=dict(color="#2C9281", line=dict(width=0)),
                            hovertemplate="%{y}<br>Split importance: %{x:,.0f}<extra></extra>",
                        )
                    )
                    plotly_layout(importance_fig, 390)
                    importance_fig.update_layout(margin=dict(l=8, r=14, t=4, b=25), showlegend=False)
                    importance_fig.update_xaxes(title="Split importance", showgrid=True, gridcolor="#E7EFEC")
                    importance_fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
                    render_plot(importance_fig, "feature_importance_chart")
                elif IMPORTANCE_IMAGE_PATH.exists():
                    st.image(str(IMPORTANCE_IMAGE_PATH), width="stretch")
                else:
                    st.caption("Feature-importance data was not found in the results folder.")

        with distribution_col:
            with st.container(border=True):
                st.markdown('<div class="tc-panel-title"><h3>Synthetic label mix</h3></div>', unsafe_allow_html=True)
                st.caption("Composition of the generated dataset—not real-world thyroid prevalence.")
                healthy_count = int(metrics.get("healthy_label_rows", 0) or 0)
                nonhealthy_count = int(metrics.get("non_healthy_phenotype_or_risk_rows", 0) or 0)
                if healthy_count + nonhealthy_count > 0:
                    dist_fig = go.Figure(
                        go.Pie(
                            labels=["Healthy-like label", "Non-Healthy / risk label"],
                            values=[healthy_count, nonhealthy_count],
                            hole=.72, sort=False,
                            marker=dict(colors=["#AFCFC7", "#1D8173"], line=dict(color="#FFFFFF", width=3)),
                            textinfo="label+percent",
                            hovertemplate="%{label}<br>Records: %{value:,}<br>Share: %{percent}<extra></extra>",
                        )
                    )
                    total = healthy_count + nonhealthy_count
                    dist_fig.add_annotation(text=f"{total:,}<br><span style='font-size:11px'>records</span>", x=.5, y=.5, showarrow=False, font=dict(size=18, color="#173D37"))
                    plotly_layout(dist_fig, 340)
                    dist_fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=8, b=5))
                    render_plot(dist_fig, "synthetic_label_mix")
                    st.caption(f"Healthy-like: {healthy_count:,} · Non-Healthy / risk: {nonhealthy_count:,}")
                else:
                    st.caption("Class counts were not recorded in the metrics file.")

        with st.expander("How to read these measures"):
            st.markdown(
                "- **Accuracy:** fraction of all records classified correctly.\n"
                "- **Balanced accuracy:** average recall across the two classes, useful when class counts differ.\n"
                "- **Precision:** among records labelled non-Healthy by the model, how many carry that synthetic label.\n"
                "- **Recall:** among records with a non-Healthy synthetic label, how many the model identified.\n"
                "- **F1-score:** combined precision/recall measure.\n"
                "- **Specificity:** proportion of Healthy-labelled records the model leaves in the Healthy-like class.\n"
                "- **ROC-AUC:** ranking performance across decision thresholds; it is not a measure of calibration."
            )


# ── Project information, methodology, and feedback ────────────────────────────
with tab_about:
    st.markdown(
        '<div class="tc-section-title"><span class="tc-step">04</span><div><h3>Project, method & limitations</h3><p>What the prototype does, how it was built, and what its results mean.</p></div></div>',
        unsafe_allow_html=True,
    )
    overview_col, method_col = st.columns(2)
    with overview_col:
        with st.container(border=True):
            st.markdown("### Project objective")
            st.write(
                "Build a user-friendly, symptom-first classification demonstration that accepts everyday answers "
                "rather than requiring users to enter thyroid lab values. It compares those answers with synthetic "
                "patterns learned by a machine-learning classifier."
            )
            st.markdown("### What the app can do")
            st.write(
                "- Collect self-reported symptoms and optional family history.\n"
                "- Show the model's pattern label and a readable summary of the user's answers.\n"
                "- Display evaluation metrics, a confusion matrix, metric profile, feature importance and label composition.\n"
                "- Export a text summary of the current check."
            )
    with method_col:
        with st.container(border=True):
            st.markdown("### Method at a glance")
            st.write("**Approach:** supervised binary classification with LightGBM.")
            st.write("**Dataset:** Synthetic Thyroid Healthcare Dataset, 1,000,000 generated rows and 125 source columns.")
            st.write("**Inputs:** age, gender, five family-history indicators and 15 self-reported symptoms.")
            st.write("**Target:** Healthy label versus any other synthetic phenotype/risk label.")
            st.write("**Split:** 700,000 train · 150,000 validation · 150,000 test records.")
            st.write("**Stack:** Python, pandas, scikit-learn, LightGBM, Streamlit and Plotly.")
            st.markdown("[Dataset source page](https://data.mendeley.com/datasets/tknsjcphcj/1)")

    st.markdown("### Project workflow")
    st.markdown(
        '<div class="tc-flow"><span>Define problem</span><b>→</b><span>Inspect & prepare data</span><b>→</b><span>Train LightGBM</span><b>→</b><span>Tune on validation</span><b>→</b><span>Evaluate</span><b>→</b><span>Streamlit UI</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Limitations to state in your presentation")
    st.warning(
        "All records are synthetic. The positive class includes non-Healthy phenotype and risk labels, "
        "not only confirmed thyroid disease. High scores on this dataset do not establish medical validity. "
        "The test split has been reused for several development comparisons, so these scores should be treated "
        "as development metrics; a fresh independent evaluation is needed for a final benchmark."
    )
    with st.expander("What has been excluded and why"):
        st.write(
            "Laboratory values (such as TSH, T3 and T4), identifiers, treatment/follow-up information and target-derived "
            "fields are excluded from the user-answerable feature set. Tree models do not require numeric feature scaling; "
            "LightGBM handles numeric missing values and the gender encoder ignores previously unseen categories. "
            "The current training workflow removes rows without a target label, but does not explicitly deduplicate all source rows."
        )

    st.markdown("### Help improve the interface")
    st.caption("This feedback is saved locally to `results/ux_feedback.csv`. Please do not include names, personal details or health information.")
    with st.form("ux_feedback_form"):
        rating = st.radio(
            "How easy was the application to understand?",
            options=[1, 2, 3, 4, 5],
            index=3,
            horizontal=True,
            format_func=lambda value: {1: "1 · Hard", 2: "2", 3: "3", 4: "4", 5: "5 · Easy"}[value],
        )
        comment = st.text_area("What should be clearer or easier? (optional)", max_chars=500)
        feedback_submitted = st.form_submit_button("Save interface feedback", width="stretch")
    if feedback_submitted:
        try:
            save_feedback(int(rating), comment)
            st.session_state["feedback_sent"] = True
        except OSError as exc:
            st.error(f"Feedback could not be saved to the project folder: {exc}")
    if st.session_state.get("feedback_sent"):
        st.success("Thanks—your interface feedback was saved locally. Symptom answers were not saved in the feedback log.")

st.markdown(
    '<div class="tc-footer">Thyroid Companion · AI Applications Lab · Educational prototype only · Not for diagnosis or treatment decisions</div>',
    unsafe_allow_html=True,
)
