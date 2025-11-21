import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).resolve().parents[1]

    raw_csv = base_dir / "data" / "friends.csv"
    fixed_csv = base_dir / "data" / "friends_preprocessed.csv"

    output_graphml = base_dir / "data" / "character_network.graphml"
    output_png = base_dir / "data" / "character_network.png"

    print(f"Loading raw dialogues: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)

    print(f"Loading emotion-fixed file: {fixed_csv}")
    df_fixed = pd.read_csv(fixed_csv)

    # ----------------------------
    # REQUIRED COLUMNS VALIDATION
    # ----------------------------
    required_raw = {"dialogue_id", "turn_id", "speaker", "utterance", "emotion"}
    required_fixed = {"dialogue_id", "turn_id", "emotion_fixed"}

    missing_raw = required_raw - set(df_raw.columns)
    missing_fixed = required_fixed - set(df_fixed.columns)

    if missing_raw:
        raise ValueError(f"Raw friends.csv missing columns: {missing_raw}")
    if missing_fixed:
        raise ValueError(f"friends_preprocessed.csv missing columns: {missing_fixed}")

    # ------------------------------------------------------
    # FIX MERGE ISSUE – REMOVE DUPLICATES IN PREPROCESSED
    # ------------------------------------------------------
    df_fixed = df_fixed.drop_duplicates(subset=["dialogue_id", "turn_id"], keep="first")

    # ------------------------------------------------------
    # MERGE RAW + PREPROCESSED SAFELY
    # ------------------------------------------------------
    df = df_raw.merge(
        df_fixed[["dialogue_id", "turn_id", "emotion_fixed"]],
        on=["dialogue_id", "turn_id"],
        how="left"
    )

    # Use fixed emotion when available
    df["emotion_final"] = df["emotion_fixed"].fillna(df["emotion"])

    df["turn_id"] = df["turn_id"].astype(int)
    df = df.sort_values(by=["dialogue_id", "turn_id"]).reset_index(drop=True)

    # ------------------------------------------------------
    # MAIN CAST FILTER
    # ------------------------------------------------------
    main_cast = {
        "Ross", "Rachel", "Joey", "Chandler", "Monica", "Phoebe",
        "Gunther", "Janice"
    }

    print("Building directed character network from dialogues...")

    G = nx.DiGraph()
    for ch in main_cast:
        G.add_node(ch)

    # ------------------------------------------------------
    # EMOTION POLARITY MAPPING
    # ------------------------------------------------------
    emotion_values = {
        "anger": -2.0,
        "disgust": -1.5,
        "fear": -1.0,
        "sadness": -1.0,
        "neutral": 0.0,
        "surprise": 0.5,
        "joy": 1.0,
    }

    # Aggregators for edges
    emotion_count = defaultdict(lambda: defaultdict(int))
    emotion_score = defaultdict(list)

    # ------------------------------------------------------
    # BUILD EDGES BASED ON TURN ORDER
    # ------------------------------------------------------
    for dlg_id, group in df.groupby("dialogue_id"):
        group = group.sort_values("turn_id")

        for i in range(len(group) - 1):
            src = group.iloc[i]
            tgt = group.iloc[i + 1]

            u = str(src["speaker"])
            v = str(tgt["speaker"])

            # Keep only main cast interactions
            if u not in main_cast or v not in main_cast:
                continue
            if u == v:
                continue

            emo = str(tgt["emotion_final"]).strip().lower()

            # Weighted interaction
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

            # Emotion counters
            emotion_count[(u, v)][emo] += 1

            if emo in emotion_values:
                emotion_score[(u, v)].append(emotion_values[emo])

    # ------------------------------------------------------
    # COMPUTE EMOTION DISTRIBUTION + DOMINANT EMOTION PER EDGE
    # ------------------------------------------------------
    print("Computing edge emotion metrics...")

    all_weights = []

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        all_weights.append(w)

        dist = emotion_count[(u, v)]
        total = sum(dist.values())

        emo_norm = {e: round(c / total, 3) for e, c in dist.items()} if total > 0 else {}

        scores = emotion_score[(u, v)]
        avg_pol = round(sum(scores) / len(scores), 3) if scores else 0.0

        dominant = max(dist.items(), key=lambda x: x[1])[0] if dist else "neutral"

        G[u][v]["emotion_distribution"] = str(emo_norm)
        G[u][v]["avg_emotion_polarity"] = avg_pol
        G[u][v]["dominant_emotion"] = dominant

    # ------------------------------------------------------
    # SKIP IF NO EDGES
    # ------------------------------------------------------
    if not G.edges():
        print("No connections inside main cast.")
        return

    # ------------------------------------------------------
    # SAVE GRAPHML
    # ------------------------------------------------------
    print("Saving GraphML...")
    nx.write_graphml(G, output_graphml)
    print(f"Saved → {output_graphml}")
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

    # ------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------
    print("Generating visualization...")

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42, k=1.1)

    # Edge color map
    emotion_colors = {
        "joy": "gold",
        "sadness": "royalblue",
        "anger": "red",
        "disgust": "brown",
        "fear": "purple",
        "surprise": "orange",
        "neutral": "gray",
    }

    # Normalize edge widths
    min_w = min(all_weights)
    max_w = max(all_weights)
    span = max_w - min_w if max_w != min_w else 1

    widths = []
    colors = []

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        norm = (w - min_w) / span
        widths.append(1 + 5 * norm)

        emo = d["dominant_emotion"]
        colors.append(emotion_colors.get(emo, "gray"))

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=3000)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold")

    plt.title("Main Cast Character Interaction Network\nEdge Color = Dominant Emotion | Edge Width = Interaction Strength")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Saved PNG → {output_png}")


if __name__ == "__main__":
    main()
