from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

base_dir = Path(__file__).resolve().parents[1]
DATA_DIR = base_dir / "data"
PAIRS_CSV = DATA_DIR / "friends_pairs_balanced.csv"
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
OUTPUT_DIR = base_dir / "models" / "emotion_influence"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LENGTH = 128
RANDOM_SEED = 42

print(f"Loading pairs dataset: {PAIRS_CSV}")
df = pd.read_csv(PAIRS_CSV)

expected_cols = {"src_utterance", "src_emotion", "tgt_emotion"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in pairs CSV: {missing}")

df = df.dropna(subset=["src_utterance", "src_emotion", "tgt_emotion"])
df = df[
    (df["src_utterance"].astype(str).str.strip() != "") &
    (df["tgt_emotion"].astype(str).str.strip() != "")
]

print(f"Total pairs after cleaning: {len(df)}")

def build_input_text(row):
    emo = str(row["src_emotion"]).strip().lower()
    utt = str(row["src_utterance"]).strip()
    return f"[SRC_EMO={emo}] {utt}"

df["input_text"] = df.apply(build_input_text, axis=1)

le = LabelEncoder()
df["label"] = le.fit_transform(df["tgt_emotion"].astype(str))
num_labels = len(le.classes_)

print("Target emotion classes:")
for i, c in enumerate(le.classes_):
    print(f"{i}: {c}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
label_map_path = OUTPUT_DIR / "label_mapping.txt"
with open(label_map_path, "w", encoding="utf-8") as f:
    for i, c in enumerate(le.classes_):
        f.write(f"{i}\t{c}\n")
print(f"Saved label mapping to: {label_map_path}")

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_df["label"]
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = Dataset.from_pandas(train_df[["input_text", "label"]])
val_ds = Dataset.from_pandas(val_df[["input_text", "label"]])
test_ds = Dataset.from_pandas(test_df[["input_text", "label"]])

train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds = val_ds.map(tokenize_batch, batched=True)
test_ds = test_ds.map(tokenize_batch, batched=True)

train_ds = train_ds.rename_columns({"label": "labels"})
val_ds = val_ds.rename_columns({"label": "labels"})
test_ds = test_ds.rename_columns({"label": "labels"})

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
model.to(device)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir=str(OUTPUT_DIR / "logs"),
    logging_steps=50,
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Training complete. Saving model...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Model saved to: {OUTPUT_DIR}")

print("Evaluating on test set...")
predictions = trainer.predict(test_ds)
logits = predictions.predictions
y_true = predictions.label_ids
y_pred = np.argmax(logits, axis=-1)

report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
print("\nClassification Report (Test)")
print(report)

report_path = OUTPUT_DIR / "test_classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved test report to: {report_path}")
