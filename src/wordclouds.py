"""
Word Clouds – Fake vs. Real (leak_free text).

Saves: results/figures/wordclouds.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGURES = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGURES, exist_ok=True)

SEED = 42


def main():
    print("=" * 60)
    print("Phase 7: Word Clouds")
    print("=" * 60)

    csv_path = os.path.join(DATA, "cleaned_leak_free.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]

    fake_text = " ".join(df[df["label"] == 0]["clean_text"].values)
    real_text = " ".join(df[df["label"] == 1]["clean_text"].values)

    wc_fake = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Reds",
        max_words=150,
        random_state=SEED,
        collocations=False,
    ).generate(fake_text)

    wc_real = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Greens",
        max_words=150,
        random_state=SEED,
        collocations=False,
    ).generate(real_text)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(wc_fake, interpolation="bilinear")
    ax1.set_title("Fake News – Word Cloud", fontsize=14, fontweight="bold", color="#e74c3c")
    ax1.axis("off")

    ax2.imshow(wc_real, interpolation="bilinear")
    ax2.set_title("Real News – Word Cloud", fontsize=14, fontweight="bold", color="#2ecc71")
    ax2.axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURES, "wordclouds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    print("\nWord cloud generation complete.")


if __name__ == "__main__":
    main()
