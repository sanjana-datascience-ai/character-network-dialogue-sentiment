import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\data")
MODEL_DIR = Path(r"C:\Users\sanja\Documents\4th year\7sem\Natural Language Processing\Assignments\Individual\Implementation\character-network-dialogue-sentiment\models\emotion_influence")

PAIRS_CSV = DATA_DIR / "friends_pairs_balanced.csv"
MAX_LENGTH = 128
RANDOM_SEED = 42

df = pd.read_csv(PAIRS_CSV)
df = df.dropna(subset=["src_utterance", "src_emotion", "tgt_emotion"])
df = df[(df["src_utterance"].astype(str).str.strip() != "") &
        (df["tgt_emotion"].astype(str).str.strip() != "")]

def build_input_text(row):
    emo = str(row["src_emotion"]).strip().lower()
    utt = str(row["src_utterance"]).strip()
    return f"[SRC_EMO={emo}] {utt}"

df["input_text"] = df.apply(build_input_text, axis=1)

labels = df["tgt_emotion"].astype(str).tolist()
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
label_names = list(le.classes_)

from sklearn.model_selection import train_test_split
_, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df["tgt_emotion"]
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_df["tgt_emotion"]
)

test_texts = test_df["input_text"].tolist()
test_labels = test_df["tgt_emotion"].astype(str).tolist()
test_labels_enc = le.transform(test_labels)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

preds = []
import torch.nn.functional as F

with torch.no_grad():
    for text in test_texts:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1).cpu().item()
        preds.append(pred)

cm = confusion_matrix(test_labels_enc, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.title("Emotion Influence Model - Confusion Matrix")

output_path = MODEL_DIR / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Confusion matrix saved to: {output_path}")
