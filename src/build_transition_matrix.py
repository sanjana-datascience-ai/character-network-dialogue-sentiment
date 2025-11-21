import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\data")
PAIRS_CSV = DATA_DIR / "friends_pairs_balanced.csv"
OUTPUT_CSV = DATA_DIR / "emotion_transition_matrix.csv"

df = pd.read_csv(PAIRS_CSV)
df = df.dropna(subset=["src_emotion", "tgt_emotion"])
df = df[(df["src_emotion"].astype(str).str.strip() != "") &
        (df["tgt_emotion"].astype(str).str.strip() != "")]

ct = df.groupby(["src_emotion", "tgt_emotion"]).size().unstack(fill_value=0)

row_sums = ct.sum(axis=1)
prob = ct.div(row_sums, axis=0)

prob.to_csv(OUTPUT_CSV, index=True)
print(f"Saved emotion transition matrix to: {OUTPUT_CSV}")
print(prob)
