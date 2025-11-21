import json
import pandas as pd
from pathlib import Path

# Input & output file paths
input_file = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\data\Raw\friends.json")
csv_file = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\data\friends.csv")

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

dialogue_counter = 0  # NEW

# Flatten nested dialogues
for dialogue in data:

    turn_counter = 0  # NEW

    if isinstance(dialogue, dict):

        for utt in dialogue.values():
            if isinstance(utt, dict):
                rows.append({
                    "dialogue_id": dialogue_counter,
                    "turn_id": turn_counter,
                    "speaker": utt.get("speaker", "").strip(),
                    "utterance": utt.get("utterance", "").strip(),
                    "emotion": utt.get("emotion", "").strip(),
                    "annotation": utt.get("annotation", "").strip()
                })
                turn_counter += 1

    elif isinstance(dialogue, list):

        for utt in dialogue:
            if isinstance(utt, dict):
                rows.append({
                    "dialogue_id": dialogue_counter,
                    "turn_id": turn_counter,
                    "speaker": utt.get("speaker", "").strip(),
                    "utterance": utt.get("utterance", "").strip(),
                    "emotion": utt.get("emotion", "").strip(),
                    "annotation": utt.get("annotation", "").strip()
                })
                turn_counter += 1

    dialogue_counter += 1  # NEW

# Convert to DataFrame
df = pd.DataFrame(rows)

# Drop empty utterances
df = df[df["utterance"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

# Save to CSV
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Successfully flattened and saved {len(df)} utterances")
print(f"Output file: {csv_file}")
print("\nSample:")
print(df.head(10).to_string(index=False))
