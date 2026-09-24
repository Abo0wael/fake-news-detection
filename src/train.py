"""
Training & Evaluation for Fake News Detection.

For both naive and leak_free cleaned data:
  - Stratified 80/20 train/test split (seed 42)
  - TF-IDF (unigrams + bigrams) inside sklearn Pipeline
  - Models: Logistic Regression, LinearSVC, Multinomial Naive Bayes
  - 5-fold stratified cross-validation on train (mean ± std F1)
  - Final test: accuracy, precision, recall, F1 (per class + macro)
  - Confusion matrix figure per model per mode

Saves:
  - results/metrics.json
  - results/metrics_table.md
  - results/figures/cm_<mode>_<model>.png
  - models/best_model.joblib
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
MODELS = os.path.join(ROOT, "models")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

SEED = 42
CLASS_NAMES = ["Fake", "Real"]

# ── Model definitions ────────────────────────────────────────────────────
def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, C=1.0
        ),
        "LinearSVC": LinearSVC(
            max_iter=2000, random_state=SEED, C=1.0
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
    }


def get_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )


def plot_confusion_matrix(y_true, y_pred, mode, model_name):
    """Save a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}\n({mode} mode)", fontweight="bold")
    plt.tight_layout()
    fname = f"cm_{mode}_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIGURES, fname), dpi=150)
    plt.close(fig)
    return fname


def evaluate_mode(mode: str):
    """Train & evaluate all models for one mode. Returns metrics dict."""
    csv_path = os.path.join(DATA, f"cleaned_{mode}.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]

    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results = {}
    best_f1 = -1
    best_pipeline = None
    best_model_name = None

    for model_name, clf in get_models().items():
        print(f"\n  → {model_name}")
        pipeline = Pipeline([
            ("tfidf", get_tfidf()),
            ("clf", clf),
        ])

        # Cross-validation on train
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                    scoring="f1_macro", n_jobs=-1)
        print(f"    CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full train, predict on test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        prec_per = precision_score(y_test, y_pred, average=None).tolist()
        rec_per = recall_score(y_test, y_pred, average=None).tolist()
        f1_per = f1_score(y_test, y_pred, average=None).tolist()

        print(f"    Test Accuracy:  {acc:.4f}")
        print(f"    Test F1 (macro): {f1_macro:.4f}")

        # Confusion matrix
        cm_file = plot_confusion_matrix(y_test, y_pred, mode, model_name)
        print(f"    Saved {cm_file}")

        results[model_name] = {
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_per_class": [round(v, 4) for v in prec_per],
            "recall_per_class": [round(v, 4) for v in rec_per],
            "f1_per_class": [round(v, 4) for v in f1_per],
            "confusion_matrix_file": cm_file,
        }

        # Track best leak_free model
        if mode == "leak_free" and f1_macro > best_f1:
            best_f1 = f1_macro
            best_pipeline = pipeline
            best_model_name = model_name

    # Save best leak_free model
    if best_pipeline is not None:
        model_path = os.path.join(MODELS, "best_model.joblib")
        joblib.dump(best_pipeline, model_path)
        print(f"\n  Best leak_free model: {best_model_name} (F1={best_f1:.4f})")
        print(f"  Saved to {model_path}")
        results["_best_model"] = best_model_name

    return results, (X_train, X_test, y_train, y_test)


def write_metrics_table(all_results):
    """Write a markdown comparison table."""
    lines = ["# Model Comparison: Naive vs. Leak-Free\n"]
    lines.append("| Mode | Model | CV F1 (mean±std) | Accuracy | Precision | Recall | F1 (macro) |")
    lines.append("|------|-------|------------------|----------|-----------|--------|------------|")

    for mode in ["naive", "leak_free"]:
        for model_name, m in all_results[mode].items():
            if model_name.startswith("_"):
                continue
            lines.append(
                f"| {mode} | {model_name} | "
                f"{m['cv_f1_mean']:.4f}±{m['cv_f1_std']:.4f} | "
                f"{m['accuracy']:.4f} | "
                f"{m['precision_macro']:.4f} | "
                f"{m['recall_macro']:.4f} | "
                f"{m['f1_macro']:.4f} |"
            )

    lines.append("\n## Per-Class Metrics (Leak-Free)\n")
    lines.append("| Model | Class | Precision | Recall | F1 |")
    lines.append("|-------|-------|-----------|--------|-----|")
    for model_name, m in all_results["leak_free"].items():
        if model_name.startswith("_"):
            continue
        for i, cls in enumerate(CLASS_NAMES):
            lines.append(
                f"| {model_name} | {cls} | "
                f"{m['precision_per_class'][i]:.4f} | "
                f"{m['recall_per_class'][i]:.4f} | "
                f"{m['f1_per_class'][i]:.4f} |"
            )

    md = "\n".join(lines)
    path = os.path.join(RESULTS, "metrics_table.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSaved {path}")


def main():
    print("=" * 60)
    print("Phase 5: Training & Evaluation")
    print("=" * 60)

    all_results = {}
    split_data = {}

    for mode in ["naive", "leak_free"]:
        print(f"\n{'─' * 50}")
        print(f"Mode: {mode}")
        print(f"{'─' * 50}")
        results, data = evaluate_mode(mode)
        all_results[mode] = results
        split_data[mode] = data

    # Save full metrics JSON
    json_path = os.path.join(RESULTS, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {json_path}")

    # Save comparison table
    write_metrics_table(all_results)

    # Save split data for downstream use (explain.py)
    np.savez(
        os.path.join(DATA, "split_leak_free.npz"),
        X_train=split_data["leak_free"][0],
        X_test=split_data["leak_free"][1],
        y_train=split_data["leak_free"][2],
        y_test=split_data["leak_free"][3],
    )
    np.savez(
        os.path.join(DATA, "split_naive.npz"),
        X_train=split_data["naive"][0],
        X_test=split_data["naive"][1],
        y_train=split_data["naive"][2],
        y_test=split_data["naive"][3],
    )
    print("Saved train/test splits for downstream phases.")

    print("\nTraining & evaluation complete.")


if __name__ == "__main__":
    main()
