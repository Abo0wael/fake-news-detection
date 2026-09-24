# EDA Summary

## Class Balance

- **Fake**: 23,481
- **Real**: 21,417

## Missing Values

- title: 0
- text: 0
- subject: 0
- date: 0
- label: 0
- label_name: 0

## Empty Texts

- Fake: 630
- Real: 1
- Total: 631

## Duplicates

### Within-class (same text repeated)
- Fake: 6,026
- Real: 225

### Cross-class duplicates: 0

## Reuters Leak – `(Reuters)` in First 200 Characters

- Fake: 0.0%
- Real: 99.1%

> **This is the strongest data leakage signal.** 99% of real articles start with a
> location + `(Reuters) -` prefix; virtually no fake articles do.

## Subject Distribution (Another Leak)

### Fake
- News: 9,050
- politics: 6,841
- left-news: 4,459
- Government News: 1,570
- US_News: 783
- Middle-east: 778

### Real
- politicsNews: 11,272
- worldnews: 10,145

> The subject categories are **completely disjoint** between classes,
> meaning subject alone perfectly predicts the label.

## Source Artifacts (Fake-Only Patterns)

| Pattern | Fake | Real |
|---------|------|------|
| 21st Century Wire | 1254 (5.3%) | 0 (0.0%) |
| Featured image | 8161 (34.8%) | 0 (0.0%) |
| Getty Images | 3943 (16.8%) | 0 (0.0%) |
| pic.twitter | 3474 (14.8%) | 0 (0.0%) |
| URLs (http) | 3302 (14.1%) | 1 (0.0%) |
| Twitter handles (@) | 6118 (26.1%) | 281 (1.3%) |

## Article Length

- Fake mean: 2,547 chars, median: 2,166
- Real mean: 2,383 chars, median: 2,222
