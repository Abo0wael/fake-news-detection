"""
Exploratory Data Analysis for the Fake and Real News Dataset.

Produces:
    - results/eda_summary.md          – textual summary
    - results/figures/class_balance.png
    - results/figures/article_length_distribution.png
    - results/figures/subject_distribution.png
"""

import os, sys, io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"Fake": "#e74c3c", "Real": "#2ecc71"}


def load_data():
    """Load both CSVs and add a label column."""
    df_fake = pd.read_csv(os.path.join(DATA, "Fake.csv"))
    df_true = pd.read_csv(os.path.join(DATA, "True.csv"))
    df_fake["label"] = 0
    df_true["label"] = 1
    df_fake["label_name"] = "Fake"
    df_true["label_name"] = "Real"
    df = pd.concat([df_fake, df_true], ignore_index=True)
    return df


def eda_report(df):
    """Run all EDA checks and return a summary dict."""
    summary = {}

    # Class balance
    balance = df["label_name"].value_counts()
    summary["class_balance"] = balance.to_dict()

    # Missing values
    summary["missing_values"] = df.isnull().sum().to_dict()

    # Empty texts
    empty_mask = df["text"].fillna("").str.strip() == ""
    summary["empty_texts"] = {
        "Fake": int(empty_mask[df["label"] == 0].sum()),
        "Real": int(empty_mask[df["label"] == 1].sum()),
        "Total": int(empty_mask.sum()),
    }

    # Exact duplicates within each class (by text)
    summary["duplicates_within_class"] = {
        "Fake": int(df[df["label"] == 0]["text"].duplicated().sum()),
        "Real": int(df[df["label"] == 1]["text"].duplicated().sum()),
    }

    # Cross-class duplicates
    fake_texts = set(df.loc[df["label"] == 0, "text"].dropna())
    true_texts = set(df.loc[df["label"] == 1, "text"].dropna())
    cross = fake_texts & true_texts
    # Remove whitespace-only matches
    cross = {t for t in cross if t.strip()}
    summary["cross_class_duplicates"] = len(cross)

    # Reuters leak
    text_col = df["text"].fillna("")
    reuters_first200 = text_col.str[:200].str.contains(r"\(Reuters\)", regex=True)
    summary["reuters_in_first_200"] = {
        "Fake": f"{reuters_first200[df['label'] == 0].mean() * 100:.1f}%",
        "Real": f"{reuters_first200[df['label'] == 1].mean() * 100:.1f}%",
    }

    # Subject distribution
    summary["subjects"] = {
        "Fake": df[df["label"] == 0]["subject"].value_counts().to_dict(),
        "Real": df[df["label"] == 1]["subject"].value_counts().to_dict(),
    }

    # Source artifacts
    artifacts = {
        "21st Century Wire": r"21st Century Wire",
        "Featured image": r"Featured image",
        "Getty Images": r"Getty Images",
        "pic.twitter": r"pic\.twitter",
        "URLs (http)": r"http",
        "Twitter handles (@)": r"@\w+",
    }
    summary["source_artifacts"] = {}
    for name, pattern in artifacts.items():
        fc = text_col[df["label"] == 0].str.contains(pattern, regex=True, case=False).sum()
        tc = text_col[df["label"] == 1].str.contains(pattern, regex=True, case=False).sum()
        summary["source_artifacts"][name] = {
            "Fake": f"{fc} ({100*fc/len(df[df['label']==0]):.1f}%)",
            "Real": f"{tc} ({100*tc/len(df[df['label']==1]):.1f}%)",
        }

    # Article length stats
    df["text_length"] = df["text"].fillna("").str.len()
    summary["article_length"] = {
        "Fake_mean": int(df[df["label"] == 0]["text_length"].mean()),
        "Fake_median": int(df[df["label"] == 0]["text_length"].median()),
        "Real_mean": int(df[df["label"] == 1]["text_length"].mean()),
        "Real_median": int(df[df["label"] == 1]["text_length"].median()),
    }

    return summary


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label_name"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[COLORS[k] for k in counts.index],
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Balance", fontweight="bold")
    ax.set_ylabel("Number of Articles")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES, "class_balance.png"), dpi=150)
    plt.close(fig)


def plot_article_length(df):
    df["text_length"] = df["text"].fillna("").str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label_name, color in COLORS.items():
        subset = df[df["label_name"] == label_name]["text_length"]
        ax.hist(subset, bins=80, alpha=0.6, color=color, label=label_name, edgecolor="white")
    ax.set_title("Article Length Distribution", fontweight="bold")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_xlim(0, 20000)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES, "article_length_distribution.png"), dpi=150)
    plt.close(fig)


def plot_subject_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label_name, color) in zip(axes, COLORS.items()):
        subset = df[df["label_name"] == label_name]
        counts = subset["subject"].value_counts()
        bars = ax.barh(counts.index, counts.values, color=color, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.set_title(f"{label_name} News – Subject Distribution", fontweight="bold")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES, "subject_distribution.png"), dpi=150)
    plt.close(fig)


def write_summary(summary):
    lines = ["# EDA Summary\n"]

    lines.append("## Class Balance\n")
    for k, v in summary["class_balance"].items():
        lines.append(f"- **{k}**: {v:,}")
    lines.append("")

    lines.append("## Missing Values\n")
    for col, cnt in summary["missing_values"].items():
        lines.append(f"- {col}: {cnt}")
    lines.append("")

    lines.append("## Empty Texts\n")
    for k, v in summary["empty_texts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Duplicates\n")
    lines.append("### Within-class (same text repeated)")
    for k, v in summary["duplicates_within_class"].items():
        lines.append(f"- {k}: {v:,}")
    lines.append(f"\n### Cross-class duplicates: {summary['cross_class_duplicates']}\n")

    lines.append("## Reuters Leak – `(Reuters)` in First 200 Characters\n")
    for k, v in summary["reuters_in_first_200"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("> **This is the strongest data leakage signal.** 99% of real articles start with a")
    lines.append("> location + `(Reuters) -` prefix; virtually no fake articles do.\n")

    lines.append("## Subject Distribution (Another Leak)\n")
    for label, subjects in summary["subjects"].items():
        lines.append(f"### {label}")
        for s, c in subjects.items():
            lines.append(f"- {s}: {c:,}")
        lines.append("")
    lines.append("> The subject categories are **completely disjoint** between classes,")
    lines.append("> meaning subject alone perfectly predicts the label.\n")

    lines.append("## Source Artifacts (Fake-Only Patterns)\n")
    lines.append("| Pattern | Fake | Real |")
    lines.append("|---------|------|------|")
    for name, vals in summary["source_artifacts"].items():
        lines.append(f"| {name} | {vals['Fake']} | {vals['Real']} |")
    lines.append("")

    lines.append("## Article Length\n")
    lines.append(f"- Fake mean: {summary['article_length']['Fake_mean']:,} chars, "
                 f"median: {summary['article_length']['Fake_median']:,}")
    lines.append(f"- Real mean: {summary['article_length']['Real_mean']:,} chars, "
                 f"median: {summary['article_length']['Real_median']:,}")
    lines.append("")

    md = "\n".join(lines)
    path = os.path.join(RESULTS, "eda_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {path}")


def main():
    print("=" * 60)
    print("Phase 2: Exploratory Data Analysis")
    print("=" * 60)
    df = load_data()
    summary = eda_report(df)

    # Print key findings to console
    print(f"\nTotal articles: {len(df):,}")
    print(f"  Fake: {summary['class_balance'].get('Fake', 0):,}")
    print(f"  Real: {summary['class_balance'].get('Real', 0):,}")
    print(f"\nEmpty texts: {summary['empty_texts']}")
    print(f"Within-class duplicates: {summary['duplicates_within_class']}")
    print(f"Cross-class duplicates: {summary['cross_class_duplicates']}")
    print(f"\nReuters in first 200 chars: {summary['reuters_in_first_200']}")
    print(f"\nSubjects (Fake): {list(summary['subjects']['Fake'].keys())}")
    print(f"Subjects (Real): {list(summary['subjects']['Real'].keys())}")

    # Figures
    plot_class_balance(df)
    print("Saved class_balance.png")
    plot_article_length(df)
    print("Saved article_length_distribution.png")
    plot_subject_distribution(df)
    print("Saved subject_distribution.png")

    # Summary
    write_summary(summary)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
