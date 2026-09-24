"""
Shortcut Test – Prove data leakage with a simple experiment.

Train Logistic Regression using ONLY the first 50 characters of each
article, in both naive and leak_free modes. Two variants:

  - body (primary):     first 50 chars of the article BODY (text column only).
                        This is where the "CITY (Reuters) -" dateline lives.
  - title+text:         first 50 chars of the cleaned title + text used for
                        training. This is mostly the headline, so it measures
                        headline style rather than the dateline leak.

If the naive version achieves very high accuracy on just 50 characters,
that is clear evidence the model is exploiting source artifacts
(Reuters prefix, etc.) rather than learning about "fake news."

Saves: results/shortcut_test_results.md, results/shortcut_test_results.json
"""

import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from clean import load_raw, remove_source_artifacts, standard_preprocess

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
N_CHARS = 50


def run_shortcut_test(texts, labels):
    """Train on first-50-chars of already-cleaned texts. Return metrics dict."""
    X = pd.Series(texts).fillna("").str[:N_CHARS].values
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # Top features
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_fake_idx = coefs.argsort()[:10]
    top_real_idx = coefs.argsort()[-10:][::-1]
    top_fake = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_fake_idx]
    top_real = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_real_idx]

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "top_fake_features": top_fake,
        "top_real_features": top_real,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def body_texts(mode: str):
    """Cleaned article bodies (no title) for one mode, with that mode's dedup rules."""
    df = load_raw()
    if mode == "naive":
        df["clean_body"] = df["text"].apply(standard_preprocess)
        return df["clean_body"], df["label"]
    df["clean_body"] = df["text"].apply(lambda t: standard_preprocess(remove_source_artifacts(t)))
    df = df[df["clean_body"].str.strip().astype(bool)]
    n_labels = df.groupby("clean_body")["label"].nunique()
    df = df[~df["clean_body"].isin(set(n_labels[n_labels > 1].index))]
    df = df.drop_duplicates(subset=["clean_body"], keep="first")
    return df["clean_body"], df["label"]


def main():
    print("=" * 60)
    print("Phase 4: Shortcut Test (First-50-Char Experiment)")
    print("=" * 60)

    results = {"body": {}, "title_text": {}}
    for mode in ["naive", "leak_free"]:
        print(f"\n[{mode}] body (first {N_CHARS} chars of article text)...")
        texts, labels = body_texts(mode)
        res = run_shortcut_test(texts, labels)
        results["body"][mode] = res
        print(f"  Accuracy: {res['accuracy']:.4f}   F1 (macro): {res['f1_macro']:.4f}")
        print(f"  Top features → REAL: {[w for w, _ in res['top_real_features'][:5]]}")

        print(f"[{mode}] title+text (first {N_CHARS} chars of cleaned training text)...")
        df = pd.read_csv(os.path.join(DATA, f"cleaned_{mode}.csv"))
        res = run_shortcut_test(df["clean_text"], df["label"])
        results["title_text"][mode] = res
        print(f"  Accuracy: {res['accuracy']:.4f}   F1 (macro): {res['f1_macro']:.4f}")

    # Save markdown report
    md = ["# Shortcut Test Results\n"]
    md.append("## Experiment")
    md.append(f"Train Logistic Regression using **only the first {N_CHARS} characters** of each article.\n")
    md.append("If the model achieves high accuracy from just 50 characters, it proves the model")
    md.append("is exploiting source artifacts (like the `CITY (Reuters) -` prefix) rather than")
    md.append("learning what fake news looks like.\n")
    md.append("- **Body (primary):** first 50 chars of the article body — where the Reuters dateline is.")
    md.append("- **Title+text (secondary):** first 50 chars of the cleaned title + text — mostly the headline,")
    md.append("  so it measures headline style (e.g. `watch`, `breaking`), which cleaning does not remove.\n")

    md.append("## Results\n")
    md.append("| Variant | Mode | Accuracy | F1 (macro) | Train Size | Test Size |")
    md.append("|---------|------|----------|------------|------------|-----------|")
    for variant, label in [("body", "Body (primary)"), ("title_text", "Title+text")]:
        for mode, res in results[variant].items():
            md.append(f"| {label} | {mode} | {res['accuracy']:.4f} | {res['f1_macro']:.4f} | "
                      f"{res['n_train']:,} | {res['n_test']:,} |")
    md.append("")

    for mode, res in results["body"].items():
        md.append(f"\n### Top Features – body, {mode} mode\n")
        md.append("**Pushing toward FAKE:**\n")
        for w, c in res["top_fake_features"]:
            md.append(f"- `{w}` ({c:.4f})")
        md.append("\n**Pushing toward REAL:**\n")
        for w, c in res["top_real_features"]:
            md.append(f"- `{w}` ({c:.4f})")

    md.append("\n## Interpretation\n")
    naive_acc = results["body"]["naive"]["accuracy"]
    lf_acc = results["body"]["leak_free"]["accuracy"]
    md.append(f"- **Naive mode** achieves **{naive_acc:.1%}** accuracy from the first 50 characters of the body.")
    md.append(f"- **Leak-free mode** drops to **{lf_acc:.1%}**.")
    md.append(f"- The **{naive_acc - lf_acc:.1%} gap** shows how much the naive model gets from the")
    md.append("  Reuters dateline alone.")

    report_path = os.path.join(RESULTS, "shortcut_test_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved report to {report_path}")

    # Also save as JSON for programmatic access
    json_path = os.path.join(RESULTS, "shortcut_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON to {json_path}")

    print("\nShortcut test complete.")


if __name__ == "__main__":
    main()
