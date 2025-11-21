import pandas as pd
from pathlib import Path
from sklearn.utils import resample

# Correct preprocessed input file
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT = BASE_DIR / "data" / "friends_preprocessed.csv"
OUTPUT = BASE_DIR / "data" / "friends_pairs_balanced.csv"

print(f"Loading cleaned file:\n{INPUT}")
df = pd.read_csv(INPUT)

required_cols = {"dialogue_id", "turn_id", "speaker", "clean_utterance", "emotion_fixed"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["turn_id"] = df["turn_id"].astype(int)
df = df.sort_values(by=["dialogue_id", "turn_id"]).reset_index(drop=True)

pairs = []
print("Building conversational A → B pairs...")

for dlg_id, group in df.groupby("dialogue_id"):
    group = group.sort_values("turn_id")

    for i in range(len(group) - 1):
        src = group.iloc[i]
        tgt = group.iloc[i + 1]

        pairs.append({
            "dialogue_id": dlg_id,
            "turn_id_src": src["turn_id"],
            "turn_id_tgt": tgt["turn_id"],

            "src_speaker": src["speaker"],
            "tgt_speaker": tgt["speaker"],

            "src_utterance": src["clean_utterance"],
            "tgt_utterance": tgt["clean_utterance"],

            "src_emotion": src["emotion_fixed"],
            "tgt_emotion": tgt["emotion_fixed"]
        })

pairs_df = pd.DataFrame(pairs)

# remove empty text rows
pairs_df = pairs_df[
    (pairs_df["src_utterance"].astype(str).str.strip() != "") &
    (pairs_df["tgt_emotion"].astype(str).str.strip() != "")
]

print(f"Pairs created: {len(pairs_df)}")

# BALANCING PAIRS
print("Balancing classes by tgt_emotion...")

counts = pairs_df["tgt_emotion"].value_counts()
max_count = counts.max()

balanced = []

for emo, emo_df in pairs_df.groupby("tgt_emotion"):
    if len(emo_df) < max_count:
        up = resample(
            emo_df,
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        balanced.append(up)
    else:
        balanced.append(emo_df)

balanced_df = pd.concat(balanced).sample(frac=1, random_state=42)
balanced_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Balanced dataset saved to:\n{OUTPUT}")
print(balanced_df["tgt_emotion"].value_counts())
