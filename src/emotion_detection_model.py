import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

id2label = model.config.id2label

def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top_id = probs.argmax()
    top_emotion = id2label[top_id]
    confidence = probs[top_id]

    return top_emotion, confidence, probs, id2label

if __name__ == "__main__":
    text = "I can't believe you did this."
    emo, conf, probs, labels = predict_emotion(text)

    print("Text:", text)
    print("Predicted Emotion:", emo)
    print("Confidence:", round(conf, 4))
