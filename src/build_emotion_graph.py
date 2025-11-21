import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\data")
PAIRS_CSV = DATA_DIR / "friends_pairs_balanced.csv"
OUTPUT_CSV = DATA_DIR / "emotion_graph_edges.csv"

df = pd.read_csv(PAIRS_CSV)

df = df.dropna(subset=["src_speaker", "tgt_speaker", "src_emotion", "tgt_emotion"])
df = df[(df["src_speaker"].astype(str).str.strip() != "") &
        (df["tgt_speaker"].astype(str).str.strip() != "")]

def polarity(e):
    e = str(e).lower()
    if e in ["joy"]:
        return "positive"
    if e in ["anger", "disgust", "fear", "sadness"]:
        return "negative"
    return "neutral"

df["tgt_polarity"] = df["tgt_emotion"].apply(polarity)

agg = df.groupby(["src_speaker", "tgt_speaker"]).agg(
    interaction_count=("dialogue_id", "count"),
    dominant_tgt_emotion=("tgt_emotion", lambda x: x.value_counts().idxmax()),
    positive_responses=("tgt_polarity", lambda x: (x == "positive").sum()),
    negative_responses=("tgt_polarity", lambda x: (x == "negative").sum()),
    neutral_responses=("tgt_polarity", lambda x: (x == "neutral").sum()),
).reset_index()

agg.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved emotion graph edges to: {OUTPUT_CSV}")
print(agg.head())
