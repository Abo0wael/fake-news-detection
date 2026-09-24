"""
Explainability – Feature importance & error analysis.

For Logistic Regression in both naive and leak_free modes:
  - Top 20 features pushing toward FAKE and toward REAL (bar charts)

For the best leak_free model:
  - 5 misclassified examples with predicted probabilities

Saves:
  - results/figures/top_features_naive.png
  - results/figures/top_features_leak_free.png
  - results/error_analysis.md
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS = os.path.join(ROOT, "models")
os.makedirs(FIGURES, exist_ok=True)

SEED = 42
sns.set_theme(style="whitegrid", font_scale=1.1)


def get_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )


def plot_top_features(mode: str):
    """Train LR and plot top 20 features for FAKE and REAL side by side."""
    csv_path = os.path.join(DATA, f"cleaned_{mode}.csv")
    df = pd.read_csv(csv_path).dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]

    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", get_tfidf()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipeline.fit(X_train, y_train)

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    # Top 20 for each direction
    top_fake_idx = coefs.argsort()[:20]
    top_real_idx = coefs.argsort()[-20:][::-1]

    fake_features = [(feature_names[i], coefs[i]) for i in top_fake_idx]
    real_features = [(feature_names[i], coefs[i]) for i in top_real_idx]

    # Print top 10
    print(f"\n  [{mode}] Top 10 features → FAKE:")
    for w, c in fake_features[:10]:
        print(f"    {w:25s}  {c:+.4f}")
    print(f"  [{mode}] Top 10 features → REAL:")
    for w, c in real_features[:10]:
        print(f"    {w:25s}  {c:+.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # FAKE side
    words_f = [w for w, _ in reversed(fake_features)]
    vals_f = [c for _, c in reversed(fake_features)]
    ax1.barh(words_f, vals_f, color="#e74c3c", edgecolor="white")
    ax1.set_title(f"Top 20 Features → FAKE\n({mode} mode)", fontweight="bold")
    ax1.set_xlabel("Coefficient")

    # REAL side
    words_r = [w for w, _ in reversed(real_features)]
    vals_r = [c for _, c in reversed(real_features)]
    ax2.barh(words_r, vals_r, color="#2ecc71", edgecolor="white")
    ax2.set_title(f"Top 20 Features → REAL\n({mode} mode)", fontweight="bold")
    ax2.set_xlabel("Coefficient")

    plt.tight_layout()
    fname = f"top_features_{mode}.png"
    fig.savefig(os.path.join(FIGURES, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

    return fake_features, real_features


def error_analysis():
    """Find misclassified examples from the best leak_free model."""
    model_path = os.path.join(MODELS, "best_model.joblib")
    pipeline = joblib.load(model_path)

    # Load split data
    split = np.load(os.path.join(DATA, "split_leak_free.npz"), allow_pickle=True)
    X_test = split["X_test"]
    y_test = split["y_test"]

    y_pred = pipeline.predict(X_test)

    # Get probabilities if available
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        probs = pipeline.predict_proba(X_test)
    elif hasattr(pipeline.named_steps["clf"], "decision_function"):
        # For LinearSVC, use decision function as confidence proxy
        dec = pipeline.decision_function(X_test)
        # Normalize to [0,1] range for display
        from scipy.special import expit
        probs_pos = expit(dec)
        probs = np.column_stack([1 - probs_pos, probs_pos])
    else:
        probs = None

    # Find misclassified
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"\n  Total misclassified: {len(misclassified_idx)} / {len(y_test)}")

    if len(misclassified_idx) == 0:
        print("  No misclassified examples found.")
        return

    # Pick 5 diverse examples
    n_examples = min(5, len(misclassified_idx))
    np.random.seed(SEED)
    sample_idx = np.random.choice(misclassified_idx, n_examples, replace=False)

    lines = ["# Error Analysis – Leak-Free Model\n"]
    lines.append(f"**Total misclassified:** {len(misclassified_idx)} / {len(y_test)} "
                 f"({100 * len(misclassified_idx) / len(y_test):.2f}%)\n")
    lines.append("## Sample Misclassified Articles\n")

    class_names = ["Fake", "Real"]
    for i, idx in enumerate(sample_idx, 1):
        text = X_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        if probs is not None:
            conf = probs[idx][y_pred[idx]]
            conf_str = f" (confidence: {conf:.2%})"
        else:
            conf_str = ""

        lines.append(f"### Example {i}")
        lines.append(f"- **True label:** {true_label}")
        lines.append(f"- **Predicted:** {pred_label}{conf_str}")
        # Show first 300 chars of text
        snippet = text[:300] + ("..." if len(text) > 300 else "")
        lines.append(f"- **Text (first 300 chars):**\n")
        lines.append(f"  > {snippet}\n")

    md = "\n".join(lines)
    path = os.path.join(RESULTS, "error_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved {path}")


def main():
    print("=" * 60)
    print("Phase 6: Explainability")
    print("=" * 60)

    all_features = {}
    for mode in ["naive", "leak_free"]:
        print(f"\n{'─' * 40}")
        print(f"Mode: {mode}")
        fake_f, real_f = plot_top_features(mode)
        all_features[mode] = {"fake": fake_f[:10], "real": real_f[:10]}

    print(f"\n{'─' * 40}")
    print("Error Analysis (leak_free model)")
    error_analysis()

    print("\nExplainability phase complete.")


if __name__ == "__main__":
    main()
