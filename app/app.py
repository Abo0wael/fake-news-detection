"""
Gradio Demo – Fake News Detector.

Loads the best leak_free model, accepts a title + text, and returns:
  - Prediction (Fake / Real) with confidence
  - Top words pushing the prediction each way
  - Disclaimer about dataset scope
"""

import os, sys
import numpy as np
import joblib
import gradio as gr

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# ── Preprocessing: reuse clean.py so the demo always matches training ────
from src.clean import clean_article_leak_free


def preprocess(title: str, text: str) -> str:
    """Apply the same cleaning as leak_free mode."""
    return clean_article_leak_free(title, text)


def get_top_words(pipeline, clean_text: str, n=10):
    """Get the top words pushing toward FAKE and REAL for this input."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    # Get the TF-IDF vector for this input
    vec = tfidf.transform([clean_text])
    nonzero = vec.nonzero()[1]

    if len(nonzero) == 0:
        return [], []

    # For each nonzero feature, compute contribution = tfidf_weight * coef
    contributions = [(feature_names[i], vec[0, i] * coefs[i]) for i in nonzero]
    contributions.sort(key=lambda x: x[1])

    top_fake = [(w, round(float(c), 4)) for w, c in contributions[:n] if c < 0]
    top_real = [(w, round(float(c), 4)) for w, c in contributions[-n:][::-1] if c > 0]

    return top_fake, top_real


# ── Load model ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODELS, "best_model.joblib")
pipeline = joblib.load(MODEL_PATH)


def predict(title: str, text: str):
    """Run prediction and return formatted results."""
    clean = preprocess(title, text)

    if not clean.strip():
        return "⚠️ Please enter some text.", ""

    # Predict
    pred = pipeline.predict([clean])[0]
    label = "🔴 FAKE" if pred == 0 else "🟢 REAL"

    # Confidence
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        prob = pipeline.predict_proba([clean])[0]
        confidence = f"{max(prob) * 100:.1f}%"
    elif hasattr(pipeline.named_steps["clf"], "decision_function"):
        from scipy.special import expit
        dec = pipeline.decision_function([clean])[0]
        confidence = f"{expit(abs(dec)) * 100:.1f}%"
    else:
        confidence = "N/A"

    # Top words
    top_fake, top_real = get_top_words(pipeline, clean)
    words_fake = "\n".join([f"  • {w} ({c:+.4f})" for w, c in top_fake]) or "  (none)"
    words_real = "\n".join([f"  • {w} ({c:+.4f})" for w, c in top_real]) or "  (none)"

    result = f"**Prediction: {label}**\n**Confidence: {confidence}**"
    explanation = (
        f"**Words pushing toward FAKE:**\n{words_fake}\n\n"
        f"**Words pushing toward REAL:**\n{words_real}"
    )

    return result, explanation


# ── Gradio UI ────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ **Disclaimer:** This model was trained on a 2016–2017 US political news "
    "dataset (Kaggle Fake and Real News Dataset). It reflects patterns in that "
    "specific corpus and is **not** a general-purpose fact-checker. It should not "
    "be used as the sole basis for determining whether news is real or fake."
)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Article Title", placeholder="Enter the article title..."),
        gr.Textbox(label="Article Text", lines=10,
                   placeholder="Paste the article text here..."),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Markdown(label="Explanation – Top Contributing Words"),
    ],
    title="🔍 Fake News Detector",
    description=(
        "Enter a news article to predict whether it is **Fake** or **Real**.\n\n"
        + DISCLAIMER
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

