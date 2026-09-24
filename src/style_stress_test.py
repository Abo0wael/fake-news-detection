"""
Style Stress Test – extra analysis, NOT the main result.

After leak_free cleaning, the strongest REAL features are Reuters house-style
markers ("said", weekday names, abbreviated months, "spokesman", ...). These
are genuine writing style rather than scraping artifacts, so the main
leak_free pipeline keeps them. This experiment removes them as well to see how
much of the remaining accuracy depends on them.

Same split, TF-IDF and Logistic Regression settings as train.py.

Saves: results/style_stress_test.md, results/style_stress_test.json
"""

import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SEED = 42

# Reuters style markers, as they appear AFTER cleaning (lowercased, lemmatized)
STYLE_MARKERS = {
    # attribution verbs
    "said", "say", "saying", "told", "tell", "reporter",
    # spokespeople / official statements
    "spokesman", "spokeswoman", "spokesperson", "statement",
    # weekday names (Reuters dates events by weekday)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # abbreviated months ("Nov. 8")
    "jan", "feb", "aug", "sept", "oct", "nov", "dec",
}


def strip_markers(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STYLE_MARKERS)


def get_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000,
                                  min_df=3, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)),
    ])


def evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipeline = get_pipeline().fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "top_fake_features": [(names[i], round(float(coefs[i]), 4)) for i in order[:10]],
        "top_real_features": [(names[i], round(float(coefs[i]), 4)) for i in order[::-1][:10]],
    }


def main():
    print("=" * 60)
    print("Extra: Style Stress Test (Reuters style markers removed)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA, "cleaned_leak_free.csv")).dropna(subset=["clean_text"])
    df["stress_text"] = df["clean_text"].apply(strip_markers)
    df = df[df["stress_text"].str.strip().astype(bool)]
    print(f"  Articles: {len(df):,}")

    results = {
        "leak_free": evaluate(df["clean_text"].values, df["label"].values),
        "style_stripped": evaluate(df["stress_text"].values, df["label"].values),
        "removed_tokens": sorted(STYLE_MARKERS),
    }
    for k in ["leak_free", "style_stripped"]:
        r = results[k]
        print(f"  {k:15s} acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}")
        print(f"    FAKE: {[w for w, _ in r['top_fake_features']]}")
        print(f"    REAL: {[w for w, _ in r['top_real_features']]}")

    md = ["# Style Stress Test (extra analysis)\n"]
    md.append("Logistic Regression on leak_free text, with and without common Reuters style markers.")
    md.append("This is **not** the main result; it measures how much accuracy depends on house style.\n")
    md.append("Removed tokens: " + ", ".join(f"`{w}`" for w in sorted(STYLE_MARKERS)) + "\n")
    md.append("| Variant | Accuracy | F1 (macro) |")
    md.append("|---------|----------|------------|")
    md.append(f"| leak_free (main) | {results['leak_free']['accuracy']:.4f} | {results['leak_free']['f1_macro']:.4f} |")
    md.append(f"| leak_free + style markers removed | {results['style_stripped']['accuracy']:.4f} | {results['style_stripped']['f1_macro']:.4f} |")
    md.append("")
    r = results["style_stripped"]
    md.append("## Top features after removing style markers\n")
    md.append("| Rank | → FAKE | → REAL |")
    md.append("|------|--------|--------|")
    for i, ((wf, cf), (wr, cr)) in enumerate(zip(r["top_fake_features"], r["top_real_features"]), 1):
        md.append(f"| {i} | `{wf}` ({cf:+.2f}) | `{wr}` ({cr:+.2f}) |")

    with open(os.path.join(RESULTS, "style_stress_test.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(RESULTS, "style_stress_test.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("  Saved results/style_stress_test.md and .json")


if __name__ == "__main__":
    main()
