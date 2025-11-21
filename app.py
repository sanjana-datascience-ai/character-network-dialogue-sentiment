import os
from pathlib import Path
import json
import math
import base64
from typing import Dict, Tuple, Optional, List

import streamlit as st

# Core data / plotting
import pandas as pd
import numpy as np

# Optional heavy libs – guarded imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# Streamlit components
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

FRIENDS_CSV = DATA_DIR / "friends.csv"
FRIENDS_PREPROCESSED = DATA_DIR / "friends_preprocessed.csv"
PAIRS_BALANCED_CSV = DATA_DIR / "friends_pairs_balanced.csv"
CHAR_NETWORK_GRAPHML = DATA_DIR / "character_network.graphml"

EMO_INFLUENCE_MODEL_DIR = MODELS_DIR / "emotion_influence"
EMO_INFLUENCE_LABEL_MAP = EMO_INFLUENCE_MODEL_DIR / "label_mapping.txt"
EMO_INFLUENCE_CONFUSION_IMG = EMO_INFLUENCE_MODEL_DIR / "confusion_matrix.png"
EMO_INFLUENCE_REPORT_TXT = EMO_INFLUENCE_MODEL_DIR / "test_classification_report.txt"

BACKGROUND_ABS = r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\background.jpeg"
BACKGROUND_REL = "background.jpeg"

AUTHOR_IMG_PATH = r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\src\author.jpg"

# Character image paths for circular image nodes in the network
CHARACTER_IMAGE_PATHS: Dict[str, str] = {
    "Monica": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\monica.jpeg",
    "Ross": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\ross.jpeg",
    "Chandler": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\chandler.jpeg",
    "Rachel": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\rachel.jpeg",
    "Joey": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\joey.jpeg",
    "Phoebe": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\phoebe.jpeg",
    "Janice": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\janice.jpeg",
    "Gunther": r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\gunthur.jpeg",
}

st.set_page_config(
    page_title="Understand Character Dynamics Like Never Before",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    """Injects custom CSS for layout, nav, hero, buttons, cards, and profile."""
    hero_bg_path = BACKGROUND_ABS.replace("\\", "\\\\")
    hero_bg_fallback = BACKGROUND_REL

    css = f"""
    <style>
    /* Global */
    body {{
        background-color: #050816;
        color: #f8fafc;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    }}

    .main {{
        padding-top: 0;
    }}

    /* Top navigation */
    .top-nav {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(5, 8, 22, 0.92);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        padding: 0.6rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .top-nav-title {{
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #e5e7eb;
    }}

    .top-nav-links a {{
        margin-left: 1.3rem;
        text-decoration: none;
        color: #cbd5f5;
        font-size: 0.95rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }}

    .top-nav-links a:hover {{
        color: #ffffff;
        border-bottom-color: #6366f1;
    }}

    .top-nav-links a.active {{
        color: #f9fafb;
        border-bottom-color: #a855f7;
        font-weight: 600;
    }}

    /* Hero section */
    .hero-container {{
        position: relative;
        min-height: 86vh;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 25px 70px rgba(15, 23, 42, 0.9);
        background-image: url("{hero_bg_path}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .hero-overlay {{
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top, rgba(15, 23, 42, 0.2), rgba(15, 23, 42, 0.92));
    }}

    .hero-content {{
        position: relative;
        z-index: 1;
        max-width: 900px;
        text-align: center;
        padding: 2.5rem 2rem 3rem 2rem;
    }}

    .hero-title {{
        font-size: clamp(2.7rem, 4vw, 3.6rem);
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        color: #f9fafb;
        text-shadow: 0 12px 35px rgba(15, 23, 42, 1);
    }}

    .hero-gradient-word {{
        background: linear-gradient(135deg, #22d3ee, #a855f7, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .hero-subtitle {{
        font-size: 1.05rem;
        color: #e2e8f0;
        max-width: 650px;
        margin: 0 auto 2rem auto;
        line-height: 1.7;
    }}

    .hero-cta-row {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }}

    .cta-button-primary, .cta-button-secondary {{
        border-radius: 999px;
        padding: 0.85rem 1.7rem;
        border: none;
        cursor: pointer;
        font-size: 0.98rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        transition: transform 0.12s ease-out, box-shadow 0.12s ease-out, background 0.15s ease-out;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.45);
    }}

    .cta-button-primary {{
        background: linear-gradient(135deg, #4f46e5, #a855f7);
        color: white;
    }}

    .cta-button-primary:hover {{
        transform: translateY(-1px);
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.65);
        background: linear-gradient(135deg, #6366f1, #c084fc);
    }}

    .cta-button-secondary {{
        background: rgba(15, 23, 42, 0.85);
        color: #e5e7eb;
        border: 1px solid rgba(148, 163, 184, 0.55);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.9);
    }}

    .cta-button-secondary:hover {{
        transform: translateY(-1px);
        background: rgba(15, 23, 42, 0.98);
        border-color: #6366f1;
    }}

    /* Cards / Panels */
    .card {{
        background: radial-gradient(circle at top left, rgba(148, 163, 184, 0.2), rgba(15, 23, 42, 0.95));
        border-radius: 1.2rem;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.35);
        margin-bottom: 1rem;
    }}

    .card-title {{
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #e5e7eb;
    }}

    .card-caption {{
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }}

    /* About profile */
    .profile-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 2rem;
    }}

    .profile-pic {{
        width: 170px;
        height: 170px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 18px 55px rgba(15, 23, 42, 0.9);
        border: 3px solid rgba(129, 140, 248, 0.8);
        margin-bottom: 1.2rem;
    }}

    .profile-name {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    .profile-role {{
        font-size: 1rem;
        color: #cbd5f5;
        margin-bottom: 0.1rem;
    }}

    .profile-institution {{
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }}

    .profile-links a {{
        color: #a5b4fc;
        margin: 0 0.6rem;
        font-size: 0.9rem;
        text-decoration: none;
    }}

    .profile-links a:hover {{
        color: #e5e7eb;
        text-decoration: underline;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_nav(current_page: str) -> None:
    """Render the custom top navigation bar."""
    pages = ["Home", "Dashboard", "Emotion Influence Prediction", "About Us"]

    links_html = []
    for p in pages:
        cls = "active" if p == current_page else ""
        links_html.append(f'<a class="{cls}" href="?page={p.replace(" ", "%20")}">{p}</a>')

    nav_html = f"""
    <div class="top-nav">
        <div class="top-nav-title">EmoDynamics</div>
        <div class="top-nav-links">
            {''.join(links_html)}
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_friends() -> Optional[pd.DataFrame]:
    if not FRIENDS_CSV.exists():
        return None
    df = pd.read_csv(FRIENDS_CSV)
    return df


@st.cache_data(show_spinner=False)
def load_preprocessed() -> Optional[pd.DataFrame]:
    if not FRIENDS_PREPROCESSED.exists():
        return None
    df = pd.read_csv(FRIENDS_PREPROCESSED)
    return df


@st.cache_data(show_spinner=False)
def load_pairs_balanced() -> Optional[pd.DataFrame]:
    if not PAIRS_BALANCED_CSV.exists():
        return None
    df = pd.read_csv(PAIRS_BALANCED_CSV)
    return df


@st.cache_data(show_spinner=False)
def load_character_graph() -> Optional["nx.DiGraph"]:
    if not HAS_NX or not CHAR_NETWORK_GRAPHML.exists():
        return None
    try:
        G = nx.read_graphml(CHAR_NETWORK_GRAPHML)
        return G
    except Exception:
        return None

EMOTION_ORDER = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]


def simple_rule_based_emotion(text: str) -> Tuple[str, Dict[str, float]]:
    """Fallback rule-based emotion detector for when no model available."""
    text_l = text.lower()
    kw = {
        "joy": ["happy", "great", "amazing", "love", "awesome", "excited"],
        "sadness": ["sad", "upset", "unhappy", "depressed", "cry", "tears"],
        "anger": ["angry", "mad", "furious", "hate", "annoyed"],
        "fear": ["scared", "afraid", "terrified", "worried", "anxious"],
        "surprise": ["surprised", "shocked", "wow", "unbelievable"],
    }
    scores = {e: 0 for e in EMOTION_ORDER}
    for emo, words in kw.items():
        for w in words:
            if w in text_l:
                scores[emo] += 1
    if all(v == 0 for v in scores.values()):
        scores["neutral"] = 1
    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}
    pred = max(probs.items(), key=lambda x: x[1])[0]
    return pred, probs


@st.cache_resource(show_spinner=False)
def load_detection_model():
    """
    Load the pretrained emotion detection model:
    j-hartmann/emotion-english-distilroberta-base
    """
    if not HAS_TRANSFORMERS:
        return None, None, None
    try:
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        id2label = getattr(model.config, "id2label", {i: str(i) for i in range(model.config.num_labels)})
        return tokenizer, model, id2label
    except Exception:
        return None, None, None


def detect_emotion(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Detect emotion using j-hartmann/emotion-english-distilroberta-base if available,
    otherwise use rule-based fallback.
    Returns: (predicted_label, {label: prob, ...})
    """
    tokenizer, model, id2label = load_detection_model()
    if tokenizer is not None and model is not None:
        try:
            device = "cuda" if (torch.cuda.is_available() and not os.environ.get("CPU_ONLY")) else "cpu"
            model.to(device)
            encoded = tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            probs_dict: Dict[str, float] = {}
            for idx, p in enumerate(probs):
                raw_label = id2label.get(idx, str(idx))
                # Map to lowercase to stay consistent
                label = str(raw_label).lower()
                probs_dict[label] = float(p)

            pred_label, _ = max(probs_dict.items(), key=lambda x: x[1])
            return pred_label, probs_dict
        except Exception:
            pass

    # Fallback
    return simple_rule_based_emotion(text)

@st.cache_resource(show_spinner=False)
def load_influence_model():
    """
    Load the trained emotion influence model from local folder, if available.
    Returns (tokenizer, model, label_id_to_name) or (None, None, None).
    """
    if not HAS_TRANSFORMERS or not EMO_INFLUENCE_MODEL_DIR.exists():
        return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(EMO_INFLUENCE_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(EMO_INFLUENCE_MODEL_DIR)
        )
        model.eval()

        label_map = {}
        if EMO_INFLUENCE_LABEL_MAP.exists():
            with open(EMO_INFLUENCE_LABEL_MAP, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        idx, lab = parts
                        label_map[int(idx)] = lab

        return tokenizer, model, label_map
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False)
def compute_empirical_transition_matrix() -> Optional[pd.DataFrame]:
    """Compute empirical transition matrix P(tgt_emotion | src_emotion) from pairs_balanced."""
    df_pairs = load_pairs_balanced()
    if df_pairs is None:
        return None
    if "src_emotion" not in df_pairs.columns or "tgt_emotion" not in df_pairs.columns:
        return None

    ct = pd.crosstab(df_pairs["src_emotion"], df_pairs["tgt_emotion"]).astype(float)
    prob = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    # reorder columns/rows by EMOTION_ORDER if present
    common = [e for e in EMOTION_ORDER if e in prob.columns]
    prob = prob.reindex(index=common, columns=common, fill_value=0)
    return prob


def predict_next_emotion(
    input_text: str, src_emotion: str
) -> Tuple[str, float, Dict[str, float], str]:
    """
    Predict next emotion given input text and detected src emotion.
    Uses trained influence model if available; otherwise empirical transitions.
    Returns: (pred_label, pred_prob, probs_dict, backend_used)
    """
    transition_matrix = compute_empirical_transition_matrix()
    tokenizer, model, label_map = load_influence_model()

    if model is not None and tokenizer is not None and label_map:
        try:
            device = "cuda" if (torch.cuda.is_available() and not os.environ.get("CPU_ONLY")) else "cpu"
            model.to(device)

            inp = f"[SRC_EMO={src_emotion.lower()}] {input_text.strip()}"
            encoded = tokenizer(
                inp,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(**encoded).logits
                probs_tensor = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            probs_dict = {}
            for idx, p in enumerate(probs_tensor):
                label = label_map.get(idx, str(idx))
                probs_dict[label] = float(p)

            pred_label, pred_prob = max(probs_dict.items(), key=lambda x: x[1])
            return pred_label, pred_prob, probs_dict, "model"
        except Exception:
            pass

    # Fallback: empirical transition from src_emotion
    if transition_matrix is not None and src_emotion in transition_matrix.index:
        row = transition_matrix.loc[src_emotion]
        probs_dict = row.to_dict()
        if not probs_dict:
            return "neutral", 1.0, {"neutral": 1.0}, "empirical"
        pred_label, pred_prob = max(probs_dict.items(), key=lambda x: x[1])
        return pred_label, float(pred_prob), probs_dict, "empirical"

    return "neutral", 1.0, {"neutral": 1.0}, "none"

def plot_emotion_distribution(df: pd.DataFrame, col: str = "emotion"):
    if not HAS_PLOTLY:
        st.warning("Plotly is not available. Install plotly to see interactive charts.")
        return
    if col not in df.columns:
        st.info(f"Column '{col}' not found.")
        return

    counts = df[col].value_counts().sort_index()
    fig_bar = px.bar(
        counts,
        x=counts.index,
        y=counts.values,
        labels={"x": "Emotion", "y": "Count"},
        title="Emotion Distribution (Counts)",
    )
    fig_bar.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        title="Emotion Distribution (Proportions)",
    )
    fig_pie.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)


def plot_utterances_per_speaker(df: pd.DataFrame):
    if not HAS_PLOTLY:
        return
    if "speaker" not in df.columns:
        return
    counts = df["speaker"].value_counts().sort_values(ascending=False).head(30)
    fig = px.bar(
        counts,
        x=counts.index,
        y=counts.values,
        labels={"x": "Speaker", "y": "Utterances"},
        title="Top Speakers by Number of Utterances",
    )
    fig.update_layout(template="plotly_dark", height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_dialogue_length_distribution(df: pd.DataFrame):
    if not HAS_PLOTLY:
        return
    if "dialogue_id" not in df.columns:
        return
    lengths = df.groupby("dialogue_id")["turn_id"].nunique()
    fig = px.histogram(
        lengths,
        nbins=40,
        labels={"value": "Dialogue Length (turns)"},
        title="Dialogue Length Distribution",
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_emotion_per_speaker(df: pd.DataFrame, emotion_col: str = "emotion"):
    if not HAS_PLOTLY:
        return
    if "speaker" not in df.columns or emotion_col not in df.columns:
        return
    pivot = pd.crosstab(df["speaker"], df[emotion_col])
    # stacked
    fig = px.bar(
        pivot,
        x=pivot.index,
        y=pivot.columns,
        title=f"Emotion Distribution per Speaker ({emotion_col})",
    )
    fig.update_layout(template="plotly_dark", height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # 100% stacked
    pivot_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig2 = px.bar(
        pivot_norm,
        x=pivot_norm.index,
        y=pivot_norm.columns,
        title=f"Emotion Proportions per Speaker ({emotion_col})",
    )
    fig2.update_layout(template="plotly_dark", height=500, xaxis_tickangle=-45, yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig2, use_container_width=True)


def plot_top_speakers_per_emotion(df: pd.DataFrame, emotion_col: str = "emotion"):
    if not HAS_PLOTLY:
        return
    if "speaker" not in df.columns or emotion_col not in df.columns:
        return

    emotions = sorted(df[emotion_col].dropna().unique())
    top_n = 5
    for emo in emotions:
        sub = df[df[emotion_col] == emo]
        counts = sub["speaker"].value_counts().head(top_n)
        if counts.empty:
            continue
        fig = px.bar(
            counts,
            x=counts.index,
            y=counts.values,
            labels={"x": "Speaker", "y": "Count"},
            title=f"Top {top_n} Speakers expressing '{emo}'",
        )
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)


def plot_emotion_timeline(df: pd.DataFrame):
    if not HAS_PLOTLY:
        return
    if "dialogue_id" not in df.columns or "turn_id" not in df.columns or "emotion" not in df.columns:
        st.info("Required columns for timeline not found.")
        return

    dialogue_ids = sorted(df["dialogue_id"].unique())
    if not dialogue_ids:
        st.info("No dialogues available.")
        return

    selected = st.selectbox("Select a dialogue_id for emotion timeline", dialogue_ids)
    sub = df[df["dialogue_id"] == selected].sort_values("turn_id")
    if sub.empty:
        st.info("No data for selected dialogue.")
        return

    emo_categories = sub["emotion"].astype(str)
    emo_to_idx = {e: i for i, e in enumerate(sorted(emo_categories.unique()))}
    sub = sub.assign(emotion_idx=emo_categories.map(emo_to_idx))

    fig = px.scatter(
        sub,
        x="turn_id",
        y="emotion_idx",
        color="emotion",
        hover_data=["speaker", "utterance"],
        title=f"Emotion Timeline for dialogue_id {selected}",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(emo_to_idx.values()),
        ticktext=list(emo_to_idx.keys()),
    )
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)


def generate_wordclouds(df: pd.DataFrame, emotion_col: str = "emotion"):
    if not HAS_WORDCLOUD or not HAS_MPL:
        st.info("Install 'wordcloud' and 'matplotlib' to see word cloud visualizations.")
        return
    if "utterance" not in df.columns or emotion_col not in df.columns:
        return

    emotions = sorted(df[emotion_col].dropna().unique())
    for emo in emotions:
        sub = df[df[emotion_col] == emo]
        text = " ".join(sub["utterance"].astype(str).tolist())
        if not text.strip():
            continue
        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="viridis",
        ).generate(text)
        st.markdown(f"#### Word Cloud for {emo}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)


def plot_ngrams(df: pd.DataFrame, emotion_col: str = "emotion", ngram_range=(2, 2), top_k: int = 10):
    if not HAS_SKLEARN or not HAS_PLOTLY:
        st.info("Install 'scikit-learn' and 'plotly' to see n-gram charts.")
        return
    if "utterance" not in df.columns or emotion_col not in df.columns:
        return

    emotions = sorted(df[emotion_col].dropna().unique())
    for emo in emotions:
        sub = df[df[emotion_col] == emo]
        text = sub["utterance"].astype(str).tolist()
        if not text:
            continue
        vec = CountVectorizer(ngram_range=ngram_range, stop_words="english")
        try:
            X = vec.fit_transform(text)
        except ValueError:
            continue
        sums = np.asarray(X.sum(axis=0)).ravel()
        freqs = list(zip(vec.get_feature_names_out(), sums))
        freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
        if not freqs:
            continue

        phrases = [f[0] for f in freqs]
        values = [f[1] for f in freqs]
        title = f"Top {top_k} {ngram_range[0]}-grams for '{emo}'"
        fig = px.bar(
            x=values[::-1],
            y=phrases[::-1],
            orientation="h",
            labels={"x": "Frequency", "y": "N-gram"},
            title=title,
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_pyvis_network(G: "nx.Graph", height: str = "650px") -> None:
    """Render a NetworkX graph using PyVis inside Streamlit (with character images where available)."""
    if not HAS_PYVIS:
        st.info("Install 'pyvis' to see the interactive character network.")
        return

    from pyvis.network import Network
    import streamlit.components.v1 as components

    net = Network(
        height=height,
        width="100%",
        bgcolor="#050816",
        font_color="#f9fafb",
        notebook=False,
        directed=True
    )
    net.barnes_hut()

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    for node in G.nodes():
        size = 25 + 25 * (degrees.get(node, 0) / max_deg)
        node_name = str(node)
        img_path = CHARACTER_IMAGE_PATHS.get(node_name)

        if img_path:
            img_b64 = load_image_base64(img_path)
            if img_b64:
                net.add_node(
                    node_name,
                    label=node_name,
                    title=node_name,
                    value=size,
                    shape="circularImage",
                    image=f"data:image/jpeg;base64,{img_b64}",
                    borderWidth=2,
                )
                continue

        net.add_node(node_name, label=node_name, title=node_name, value=size)

    emotion_colors = {
        "joy": "#facc15",
        "sadness": "#3b82f6",
        "anger": "#ef4444",
        "disgust": "#92400e",
        "fear": "#7c3aed",
        "surprise": "#f97316",
        "neutral": "#9ca3af",
    }

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        dom = str(data.get("dominant_emotion", "neutral"))
        color = emotion_colors.get(dom, "#9ca3af")
        tooltip = f"{u} → {v}<br>Weight: {w}<br>Dominant emotion: {dom}"
        net.add_edge(u, v, value=w, color=color, title=tooltip)

    tmp_path = BASE_DIR / "app" / "tmp_network.html"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(tmp_path))

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=700, scrolling=True)


def plot_adjacency_heatmap(G: "nx.DiGraph"):
    if not HAS_PLOTLY or not HAS_NX:
        return
    nodes = sorted(G.nodes())
    mat = np.zeros((len(nodes), len(nodes)))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                mat[i, j] = G[u][v].get("weight", 0)

    fig = px.imshow(
        mat,
        x=nodes,
        y=nodes,
        labels={"x": "Target", "y": "Source", "color": "Weight"},
        title="Character Interaction Adjacency Heatmap (Weight = Exchanges)",
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_centrality_bars(G: "nx.DiGraph"):
    if not HAS_PLOTLY or not HAS_NX:
        return
    if not G.nodes():
        st.info("Graph is empty.")
        return

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)

    deg_ser = pd.Series(deg).sort_values(ascending=False)
    btw_ser = pd.Series(btw).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            deg_ser,
            x=deg_ser.index,
            y=deg_ser.values,
            labels={"x": "Character", "y": "Degree Centrality"},
            title="Degree Centrality",
        )
        fig1.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(
            btw_ser,
            x=btw_ser.index,
            y=btw_ser.values,
            labels={"x": "Character", "y": "Betweenness Centrality"},
            title="Betweenness Centrality",
        )
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)


def plot_transition_heatmap(matrix: pd.DataFrame, title: str = "Emotion Transition Matrix"):
    if not HAS_PLOTLY:
        return
    fig = px.imshow(
        matrix.values,
        x=matrix.columns,
        y=matrix.index,
        labels={"x": "Next Emotion", "y": "Current Emotion", "color": "P"},
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_transition_heatmap_with_highlight(matrix: pd.DataFrame, highlight_row: str):
    if not HAS_PLOTLY:
        return
    if highlight_row not in matrix.index:
        plot_transition_heatmap(matrix, "Emotion Transition Matrix")
        return

    base_z = matrix.values
    y_labels = matrix.index.tolist()
    x_labels = matrix.columns.tolist()
    row_idx = y_labels.index(highlight_row)

    highlight_z = np.full_like(base_z, np.nan, dtype=float)
    highlight_z[row_idx, :] = base_z[row_idx, :]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=base_z,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            colorbar=dict(title="P"),
            zmin=0,
            zmax=matrix.values.max(),
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=highlight_z,
            x=x_labels,
            y=y_labels,
            colorscale="Reds",
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Emotion Transition Matrix (highlighted: {highlight_row})",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def parse_classification_report_text(report_text: str) -> Optional[pd.DataFrame]:
    """
    Parse a sklearn classification_report text into a DataFrame:
    rows = classes, cols = precision / recall / f1-score / support
    """
    lines = [l.strip() for l in report_text.splitlines() if l.strip()]
    if not lines:
        return None

    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("precision"):
            header_idx = i
            break

    if header_idx is None:
        return None

    data_rows = []
    for line in lines[header_idx + 1 :]:
        lower = line.lower()
        if lower.startswith("accuracy") or lower.startswith("macro avg") or lower.startswith("weighted avg"):
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        label = parts[0]
        try:
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(float(parts[4]))
        except ValueError:
            continue
        data_rows.append(
            {
                "emotion": label,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )

    if not data_rows:
        return None
    df = pd.DataFrame(data_rows).set_index("emotion")
    return df


def make_dummy_classification_report(labels: List[str]) -> pd.DataFrame:
    """Create a dummy zero-filled classification report DataFrame."""
    return pd.DataFrame(
        {
            "precision": [0.0] * len(labels),
            "recall": [0.0] * len(labels),
            "f1-score": [0.0] * len(labels),
            "support": [0] * len(labels),
        },
        index=labels,
    )

def render_home():
    """Render the hero landing page."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-title">
                    Understand Character Dynamics <span class="hero-gradient-word">Like Never Before</span>
                </div>
                <div class="hero-subtitle">
                    Explore how emotions ripple through conversations, how characters influence one another,
                    and how subtle shifts in tone transform a simple dialogue into a rich emotional network.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        '<div class="card"><div class="card-title">What is EmoDynamics?</div>'
        '<div class="card-caption">A character-centric NLP lab built on Friends dialogues.</div>'
        "<p>EmoDynamics lets you inspect the emotional life of a series: "
        "from low-level utterance emotions to high-level character networks and learned emotion influence patterns. "
        "Use the navigation menu above to explore insights, analytics, and prediction tools.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard():
    """Render the multi-tab analytics dashboard."""
    st.markdown("## 📊 Analytics Dashboard")
    tabs = st.tabs(["Dataset", "Character Network", "Emotion Influence", "Text Analysis"])

    df = load_friends()
    df_pre = load_preprocessed()

    with tabs[0]:
        st.markdown('<div class="card-title">Dataset Overview</div>', unsafe_allow_html=True)
        if df is None:
            st.error(f"Could not find dataset at {FRIENDS_CSV}.")
        else:
            st.markdown(
                '<div class="card-caption">friends.csv – utterance-level dialogue annotations.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(20))

            st.markdown('<div class="card-title">Global Emotion Distribution</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card-caption">Original labels (including non-neutral).</div>', unsafe_allow_html=True)
                plot_emotion_distribution(df, col="emotion")

            with col2:
                if df_pre is not None and "emotion_fixed" in df_pre.columns:
                    st.markdown('<div class="card-caption">After replacing non-neutral with predicted emotions.</div>',
                                unsafe_allow_html=True)
                    plot_emotion_distribution(df_pre, col="emotion_fixed")
                else:
                    st.markdown(
                        '<div class="card-caption">No emotion_fixed column found – showing filtered distribution without non-neutral.</div>',
                        unsafe_allow_html=True,
                    )
                    df_no_nn = df[df["emotion"] != "non-neutral"].copy()
                    plot_emotion_distribution(df_no_nn, col="emotion")

            st.markdown('<div class="card-title">Utterances per Speaker</div>', unsafe_allow_html=True)
            plot_utterances_per_speaker(df)

            st.markdown('<div class="card-title">Dialogue Length Distribution</div>', unsafe_allow_html=True)
            plot_dialogue_length_distribution(df)

            st.markdown('<div class="card-title">Emotion × Speaker Patterns</div>', unsafe_allow_html=True)
            if df_pre is not None and "emotion_fixed" in df_pre.columns:
                plot_emotion_per_speaker(df_pre, emotion_col="emotion_fixed")
            else:
                plot_emotion_per_speaker(df, emotion_col="emotion")

            st.markdown('<div class="card-title">Top Speakers per Emotion</div>', unsafe_allow_html=True)
            if df_pre is not None and "emotion_fixed" in df_pre.columns:
                plot_top_speakers_per_emotion(df_pre, emotion_col="emotion_fixed")
            else:
                plot_top_speakers_per_emotion(df, emotion_col="emotion")

            st.markdown('<div class="card-title">Emotion Timeline (per dialogue)</div>', unsafe_allow_html=True)
            plot_emotion_timeline(df)

    with tabs[1]:
        st.markdown('<div class="card-title">Character Network</div>', unsafe_allow_html=True)
        G = load_character_graph()
        if G is None:
            st.error(f"Could not load character_network.graphml at {CHAR_NETWORK_GRAPHML}.")
        else:
            st.markdown(
                '<div class="card-caption">Nodes are characters; edges are directed dialogue exchanges enriched with dominant emotion and weight.</div>',
                unsafe_allow_html=True,
            )
            st.subheader("Interactive Main Cast Network")
            render_pyvis_network(G)

            st.subheader("Adjacency Heatmap")
            plot_adjacency_heatmap(G)

            st.subheader("Centrality Metrics")
            plot_centrality_bars(G)

            st.subheader("Ego Network Explorer")
            nodes = sorted(G.nodes())
            if nodes:
                selected = st.selectbox("Select a character", nodes)
                if HAS_NX:
                    ego = nx.ego_graph(G, selected, radius=1, center=True, undirected=False)
                    st.markdown(
                        f'<div class="card-caption">Showing immediate interaction neighborhood for <b>{selected}</b>.</div>',
                        unsafe_allow_html=True,
                    )
                    render_pyvis_network(ego, height="550px")
                else:
                    st.info("NetworkX not available, cannot build ego network.")

    with tabs[2]:
        st.markdown('<div class="card-title">Emotion Influence Patterns</div>', unsafe_allow_html=True)
        matrix = compute_empirical_transition_matrix()
        if matrix is None:
            st.error("Could not compute empirical transition matrix – check friends_pairs_balanced.csv.")
        else:
            st.subheader("Empirical Transition Matrix")
            plot_transition_heatmap(matrix, "Empirical P(next emotion | current emotion)")

            st.subheader("Per-Emotion Outgoing Distributions")
            src_emos = list(matrix.index)
            if src_emos:
                selected_src = st.selectbox("Select a source emotion", src_emos, key="outgoing_src")
                row = matrix.loc[selected_src]
                if HAS_PLOTLY:
                    fig = px.bar(
                        x=row.index,
                        y=row.values,
                        labels={"x": "Next Emotion", "y": "Probability"},
                        title=f"Outgoing distribution from '{selected_src}'",
                    )
                    fig.update_layout(template="plotly_dark", height=350, yaxis=dict(tickformat=".0%"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(row)

            st.subheader("Per-Emotion Incoming Distributions")
            if matrix is not None:
                tgt_emos = list(matrix.columns)
                selected_tgt = st.selectbox("Select a target emotion", tgt_emos, key="incoming_tgt")
                col = matrix[selected_tgt]
                col = col[col > 0]
                if HAS_PLOTLY and not col.empty:
                    fig = px.bar(
                        x=col.index,
                        y=col.values,
                        labels={"x": "Source Emotion", "y": "Probability"},
                        title=f"Incoming distribution for '{selected_tgt}'",
                    )
                    fig.update_layout(template="plotly_dark", height=350, yaxis=dict(tickformat=".0%"))
                    st.plotly_chart(fig, use_container_width=True)
                elif not col.empty:
                    st.write(col)

            tokenizer, model, label_map = load_influence_model()
            if model is not None and tokenizer is not None and label_map:
                st.subheader("Model-based Transition View (sampled)")
                st.markdown(
                    "compare model-predicted transitions with empirical ones.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Trained influence model not available or transformers not installed. Using empirical transitions only.")

            st.subheader("Model Evaluation (Confusion Matrix & Report)")
            if EMO_INFLUENCE_CONFUSION_IMG.exists():
                st.markdown('<div class="card-caption">Confusion matrix from held-out test set.</div>',
                            unsafe_allow_html=True)
                st.image(str(EMO_INFLUENCE_CONFUSION_IMG), caption="Emotion Influence Model – Confusion Matrix")
            report_df = None
            if EMO_INFLUENCE_REPORT_TXT.exists():
                st.markdown('<div class="card-caption">Per-class precision / recall / F1 from classification_report.</div>',
                            unsafe_allow_html=True)
                with open(EMO_INFLUENCE_REPORT_TXT, "r", encoding="utf-8") as f:
                    report_txt = f.read()
                st.text(report_txt)
                parsed = parse_classification_report_text(report_txt)
                report_df = parsed

            st.subheader("Classification Report (Table)")
            if model is None or tokenizer is None or label_map is None:
                labels = EMOTION_ORDER
                dummy_df = make_dummy_classification_report(labels)
                st.warning(
                    "Influence model not available – showing a dummy classification report table with zeros.",
                    icon="⚠",
                )
                st.dataframe(dummy_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}))
            else:
                if report_df is None:
                    labels = sorted(set(label_map.values()))
                    report_df = make_dummy_classification_report(labels)
                    st.warning(
                        "Could not parse test_classification_report.txt – showing a zero-filled table based on model labels.",
                        icon="⚠",
                    )
                st.dataframe(report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}))

    with tabs[3]:
        st.markdown('<div class="card-title">Text-Level Analysis</div>', unsafe_allow_html=True)
        if df is None:
            st.error(f"Could not find dataset at {FRIENDS_CSV}.")
            return

        emo_col = "emotion"
        if df_pre is not None and "emotion_fixed" in df_pre.columns:
            emo_col = "emotion_fixed"
            df_for_text = df_pre.copy()
            st.markdown(
                '<div class="card-caption">Using emotion_fixed from preprocessed file for text analysis.</div>',
                unsafe_allow_html=True,
            )
        else:
            df_for_text = df.copy()

        st.subheader("Word Clouds per Emotion")
        generate_wordclouds(df_for_text, emotion_col=emo_col)

        st.subheader("Top Bigrams per Emotion")
        plot_ngrams(df_for_text, emotion_col=emo_col, ngram_range=(2, 2), top_k=10)

        st.subheader("Utterance Length vs Emotion")
        if HAS_PLOTLY:
            df_len = df_for_text.copy()
            df_len["utterance_len"] = df_len["utterance"].astype(str).str.split().apply(len)
            fig = px.box(
                df_len,
                x=emo_col,
                y="utterance_len",
                title="Token Count Distribution per Emotion",
                labels={"utterance_len": "Token Count", emo_col: "Emotion"},
            )
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install plotly for boxplot visualization.")


def render_prediction():
    """Render the Emotion Influence Prediction tool."""
    st.markdown("## 🎯 Emotion Influence Prediction")
    st.markdown(
        '<div class="card-caption">Type an utterance and estimate how the next speaker is likely to respond emotionally.</div>',
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Enter a single utterance",
        value="I can't believe this happened to me today.",
        height=120,
    )

    if st.button("Analyze Emotion Influence", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Detecting emotion and predicting likely response..."):
            emo_x, emo_probs = detect_emotion(user_text)
            emo_x_prob = emo_probs.get(emo_x, max(emo_probs.values()) if emo_probs else 1.0)

            emo_y, emo_y_prob, emo_y_probs, backend = predict_next_emotion(user_text, emo_x)

        st.markdown(
            f"*Emotion Detected for Person X:* **{emo_x}**  \n"
            f"*Predicted Reaction:*  \n"
            f"\"In reaction to this utterance, it's **{emo_y_prob*100:.1f}%** likely that the next person will respond in **{emo_y}**.\"",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Emotion Detection Probabilities (Person X)")
            if HAS_PLOTLY and emo_probs:
                fig1 = px.bar(
                    x=list(emo_probs.keys()),
                    y=list(emo_probs.values()),
                    labels={"x": "Emotion", "y": "Probability"},
                    title="Emotion Detection Probability",
                )
                fig1.update_layout(template="plotly_dark", height=350, yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write(emo_probs)
        with col2:
            st.markdown("#### Next Speaker Emotion Probabilities (Person Y)")
            backend_txt = {
                "model": "Trained influence model",
                "empirical": "Empirical transition probabilities",
                "none": "Fallback heuristic",
            }.get(backend, backend)
            st.markdown(
                f'<div class="card-caption">Backend used: {backend_txt}</div>',
                unsafe_allow_html=True,
            )

            if HAS_PLOTLY and emo_y_probs:
                emo_y_probs_sorted = dict(sorted(emo_y_probs.items(), key=lambda x: x[1], reverse=True))
                fig2 = px.bar(
                    x=list(emo_y_probs_sorted.values()),
                    y=list(emo_y_probs_sorted.keys()),
                    orientation="h",
                    labels={"x": "Probability", "y": "Next Emotion"},
                    title="Next Emotion Probability Distribution",
                )
                fig2.update_layout(template="plotly_dark", height=350, xaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write(emo_y_probs)

        matrix = compute_empirical_transition_matrix()
        if matrix is not None:
            st.markdown("#### Where does this sit in the global emotion flow?")
            plot_transition_heatmap_with_highlight(matrix, highlight_row=emo_x)


def load_image_base64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def render_about():
    """Render About Us page."""
    st.markdown("## 👩‍💻 About Us")

    img_b64 = load_image_base64(AUTHOR_IMG_PATH)

    if img_b64:
        st.markdown(
            f"""
            <div class="profile-container">
                <img src="data:image/jpeg;base64,{img_b64}" class="profile-pic" />
                <div class="profile-name">Sanjana R</div>
                <div class="profile-role">4th year Student, B. Tech (Hons) Data Science</div>
                <div class="profile-institution">Vidyashilp University</div>
                <div class="profile-links">
                    <a href="mailto:2022sanjana.r@vidyashilp.edu.in">Email</a>
                    <a href="https://www.linkedin.com/in/sanjana-ravindra-761393355?utm_source=share_via&utm_content=profile&utm_medium=member_ios" target="_blank">LinkedIn</a>
                    <a href="https://github.com/sanjana-datascience-ai" target="_blank">GitHub</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="profile-container">
                <div class="profile-name">Sanjana R</div>
                <div class="profile-role">4th year Student, B. Tech (Hons) Data Science</div>
                <div class="profile-institution">Vidyashilp University</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
        <div class="card-title">Project Vision</div>
        <p>
        This project explores how emotions propagate through dialogues in the TV show <b>Friends</b>.
        It combines NLP, graph analysis, and deep learning to model:
        </p>
        <ul>
            <li>Utterance-level emotion detection</li>
            <li>Emotion influence between speakers</li>
            <li>Character interaction networks enriched with emotional context</li>
        </ul>
        <p>
        The end goal is to provide an intuitive and interactive way to <b>understand character dynamics like never before</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    inject_css()

    params = st.query_params
    raw_page = params.get("page", "Home")

    if isinstance(raw_page, list):
        raw_page = raw_page[0]

    current_page = raw_page.replace("%20", " ")

    if current_page not in ["Home", "Dashboard", "Emotion Influence Prediction", "About Us"]:
        current_page = "Home"

    render_nav(current_page)

    if current_page == "Home":
        render_home()
    elif current_page == "Dashboard":
        render_dashboard()
    elif current_page == "Emotion Influence Prediction":
        render_prediction()
    elif current_page == "About Us":
        render_about()


if __name__ == "__main__":
    main()
