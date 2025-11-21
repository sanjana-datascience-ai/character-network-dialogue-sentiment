import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.utils import resample


def main():
    base_dir = Path(__file__).resolve().parents[1]
    input_csv = base_dir / "data" / "friends.csv"
    output_csv = base_dir / "data" / "friends_preprocessed.csv"

    print(f"Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"’", "'", text)
        text = re.sub(r"[^a-z!?'\s]", "", text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)

    def extract_spacy_features(text):
        if not isinstance(text, str) or not text.strip():
            return [], []
        doc = nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return pos_tags, entities

    print("Cleaning utterances...")
    df["clean_utterance"] = df["utterance"].apply(clean_text)

    print("Extracting POS + NER...")
    df["pos_tags"], df["entities"] = zip(*df["utterance"].apply(extract_spacy_features))

    # -----------------------------------------------------------------------------------------
    #   ONLY PREDICT EMOTION FOR ROWS WHERE emotion == "non-neutral"
    # -----------------------------------------------------------------------------------------
    print("Loading emotion classifier for fixing 'non-neutral' labels...")
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    id2label = model.config.id2label

    @torch.no_grad()
    def predict_emotion(text):
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        logits = model(**encoded).logits
        pred = logits.argmax(dim=-1).item()
        return id2label[pred]

    print("Fixing non-neutral emotions...")
    df["predicted_emotion"] = df.apply(
        lambda row: predict_emotion(row["clean_utterance"]) if row["emotion"] == "non-neutral" else row["emotion"],
        axis=1
    )

    df["emotion_fixed"] = df["predicted_emotion"]

    # -----------------------------------------------------------------------------------------
    # BALANCE CLASSES
    # -----------------------------------------------------------------------------------------
    print("Balancing class distribution...")
    max_count = df["emotion_fixed"].value_counts().max()
    emotions = df["emotion_fixed"].unique()

    balanced_df = pd.concat([
        resample(
            df[df["emotion_fixed"] == emo],
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        for emo in emotions
    ])

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep only necessary columns
    keep_cols = [
        "dialogue_id", "turn_id", "speaker",
        "utterance", "clean_utterance",
        "emotion", "emotion_fixed",
        "annotation", "pos_tags", "entities"
    ]

    balanced_df = balanced_df[keep_cols]

    print(f"Saving preprocessed dataset: {output_csv}")
    balanced_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\nSample:")
    print(balanced_df.head(10).to_string(index=False))
    print("\nPreprocessing DONE.")


if __name__ == "__main__":
    main()
