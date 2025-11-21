import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def emotion_to_polarity(e):
    mapping = {
        "anger": -2,
        "disgust": -1.5,
        "fear": -1,
        "sadness": -1,
        "neutral": 0,
        "surprise": +0.5,
        "joy": +1
    }
    return mapping.get(str(e).lower().strip(), 0)


def main():
    base = Path(__file__).resolve().parents[1]
    input_csv = base / "data" / "friends_pairs_balanced.csv"
    output_png = base / "data" / "character_influence_heatmap.png"

    print(f"Loading pairs: {input_csv}")
    df = pd.read_csv(input_csv)

    df["polarity"] = df["tgt_emotion"].apply(emotion_to_polarity)

    speakers = sorted(df["src_speaker"].unique())
    heatmap_matrix = pd.DataFrame(0.0, index=speakers, columns=speakers)

    for (a, b), group in df.groupby(["src_speaker", "tgt_speaker"]):
        avg_pol = group["polarity"].mean()
        heatmap_matrix.loc[a, b] = avg_pol

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Emotional Polarity"}
    )

    plt.title("Character Influence Heatmap (A → B Emotional Impact)")
    plt.ylabel("Speaker A (who speaks)")
    plt.xlabel("Speaker B (who responds)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved heatmap to: {output_png}")


if __name__ == "__main__":
    main()
