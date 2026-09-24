"""
Text cleaning for the Fake/Real News dataset.

Two modes:
  - "naive":     title + text, standard NLP preprocessing only.
  - "leak_free": remove all known source artifacts (Reuters dateline,
                 image credits, title tags like "(VIDEO)", tweet embeds,
                 URLs, Twitter handles, "via", ...) from title and text
                 separately, then standard preprocessing.
                 Drop cross-class duplicates, then deduplicate.

Produces:
  - data/cleaned_naive.csv
  - data/cleaned_leak_free.csv

Standard preprocessing (both modes):
  lowercase → remove punctuation & numbers → stopword removal →
  WordNet lemmatization
"""

import os, re, sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ── NLTK setup ───────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── Source-artifact patterns to remove in leak_free mode ─────────────────
# Reuters dateline: "WASHINGTON (Reuters) - ", "SEATTLE/WASHINGTON (Reuters) - ".
# Not anchored to line start, so it also works if applied after a title.
REUTERS_PREFIX_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z .,/'\-]{0,60}?\s*)?\(Reuters\)\s*[-–—]?\s*"
)
# The word "reuters" anywhere
REUTERS_WORD_RE = re.compile(r"\breuters\b", re.IGNORECASE)
# 21st Century Wire attribution
WIRE_RE = re.compile(r"21st\s*century\s*wire", re.IGNORECASE)
# Image credits and captions (trailing period optional: credits usually end the article)
IMAGE_CREDITS_RE = re.compile(
    r"(Featured image[^\n.]*(?:\.|$)|"
    r"(?:photo|image)s?\s*(?:via|by|credit|courtesy)[^\n.]*(?:\.|$)|"
    r"IMAGE\s*:[^\n.]*(?:\.|$)|"
    r"Photo\s*:[^\n.]*(?:\.|$))",
    re.IGNORECASE | re.MULTILINE,
)
# Photo/embed source names that only the fake-news sites use
MEDIA_SOURCE_RE = re.compile(
    r"getty\s*images?|screen\s*(?:shot|capture|grab)s?|flickr|youtube|shutterstock",
    re.IGNORECASE,
)
# Embedded tweet signature: "— Name (@handle) December 30, 2017"
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
TWEET_SIGNATURE_RE = re.compile(
    r"[—–-]\s*[^\n()]{0,60}\(@\w+\)\s*" + MONTHS + r"\s+\d{1,2},\s*\d{4}",
    re.IGNORECASE,
)
# Title tags like "(VIDEO)", "[TWEETS]", "(IMAGES)"
TITLE_TAG_RE = re.compile(
    r"[\(\[]\s*(?:video|videos|images?|tweets?|watch|details|photos?|screenshots?)\s*[\)\]]",
    re.IGNORECASE,
)
# Blog boilerplate: "Read more:" and "via" (as in "via Twitter", "image via ...")
READ_MORE_RE = re.compile(r"\bread more\s*:?", re.IGNORECASE)
VIA_RE = re.compile(r"\bvia\b", re.IGNORECASE)
# URLs
URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Twitter handles
TWITTER_HANDLE_RE = re.compile(r"@\w+")
# pic.twitter links
PIC_TWITTER_RE = re.compile(r"pic\.twitter\.com/\S+")
# Residual "ADVERTISEMENT" / "Advertisement"
AD_RE = re.compile(r"\bADVERTISEMENT\b", re.IGNORECASE)


def load_raw():
    """Load both CSVs, add labels, combine title + text."""
    df_fake = pd.read_csv(os.path.join(DATA, "Fake.csv"))
    df_true = pd.read_csv(os.path.join(DATA, "True.csv"))
    df_fake["label"] = 0
    df_true["label"] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    # Combine title and text
    df["combined"] = df["title"] + " " + df["text"]
    return df


def remove_source_artifacts(text: str) -> str:
    """Strip all known source/attribution artifacts from text."""
    text = REUTERS_PREFIX_RE.sub(" ", text)
    text = REUTERS_WORD_RE.sub(" ", text)
    text = WIRE_RE.sub(" ", text)
    text = TWEET_SIGNATURE_RE.sub(" ", text)   # before handles are removed
    text = IMAGE_CREDITS_RE.sub(" ", text)
    text = MEDIA_SOURCE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = PIC_TWITTER_RE.sub(" ", text)
    text = TWITTER_HANDLE_RE.sub(" ", text)
    text = AD_RE.sub(" ", text)
    text = TITLE_TAG_RE.sub(" ", text)
    text = READ_MORE_RE.sub(" ", text)
    text = VIA_RE.sub(" ", text)
    return text


def clean_article_leak_free(title: str, text: str) -> str:
    """Full leak_free cleaning of one article (used by clean.py and the demo app).

    Title and text are cleaned separately so that patterns tied to the start of
    the body (the Reuters dateline) are not hidden behind the title.
    """
    combined = (remove_source_artifacts(title or "") + " "
                + remove_source_artifacts(text or ""))
    return standard_preprocess(combined)


def standard_preprocess(text: str) -> str:
    """Lowercase, remove punctuation/numbers, stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def clean_naive(df):
    """Naive mode: combine title+text, standard preprocessing only."""
    print("  Applying standard preprocessing (naive mode)...")
    df = df.copy()
    df["clean_text"] = df["combined"].apply(standard_preprocess)
    return df[["clean_text", "label"]]


def clean_leak_free(df):
    """Leak-free mode: remove artifacts, preprocess, deduplicate."""
    df = df.copy()

    # 1-2. Remove source artifacts (title and text separately) + standard preprocessing
    print("  Removing source artifacts and applying standard preprocessing (leak_free mode)...")
    df["clean_text"] = [clean_article_leak_free(t, x) for t, x in zip(df["title"], df["text"])]

    # 3. Drop empty texts
    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)].copy()
    print(f"  Dropped {before - len(df)} empty texts after cleaning.")

    # 4. Remove articles whose cleaned text appears in BOTH classes
    #    (must run before deduplication, which would keep one copy and hide the conflict)
    dup_texts = df.groupby("clean_text")["label"].nunique()
    cross_dup_texts = set(dup_texts[dup_texts > 1].index)
    before = len(df)
    df = df[~df["clean_text"].isin(cross_dup_texts)]
    print(f"  Dropped {before - len(df)} cross-class duplicate articles.")

    # 5. Deduplicate (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"], keep="first")
    print(f"  Dropped {before - len(df)} exact duplicates.")

    return df[["clean_text", "label"]].reset_index(drop=True)


def main():
    print("=" * 60)
    print("Phase 3: Text Cleaning")
    print("=" * 60)

    df = load_raw()
    print(f"Loaded {len(df):,} articles.\n")

    # Naive
    print("[Naive mode]")
    df_naive = clean_naive(df)
    naive_path = os.path.join(DATA, "cleaned_naive.csv")
    df_naive.to_csv(naive_path, index=False)
    print(f"  Saved {naive_path}  ({len(df_naive):,} rows)\n")

    # Leak-free
    print("[Leak-free mode]")
    df_lf = clean_leak_free(df)
    lf_path = os.path.join(DATA, "cleaned_leak_free.csv")
    df_lf.to_csv(lf_path, index=False)
    print(f"  Saved {lf_path}  ({len(df_lf):,} rows)\n")

    print("Cleaning complete.")


if __name__ == "__main__":
    main()
