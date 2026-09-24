# Shortcut Test Results

## Experiment
Train Logistic Regression using **only the first 50 characters** of each article.

If the model achieves high accuracy from just 50 characters, it proves the model
is exploiting source artifacts (like the `CITY (Reuters) -` prefix) rather than
learning what fake news looks like.

- **Body (primary):** first 50 chars of the article body — where the Reuters dateline is.
- **Title+text (secondary):** first 50 chars of the cleaned title + text — mostly the headline,
  so it measures headline style (e.g. `watch`, `breaking`), which cleaning does not remove.

## Results

| Variant | Mode | Accuracy | F1 (macro) | Train Size | Test Size |
|---------|------|----------|------------|------------|-----------|
| Body (primary) | naive | 0.9971 | 0.9971 | 35,918 | 8,980 |
| Body (primary) | leak_free | 0.9000 | 0.8990 | 30,680 | 7,671 |
| Title+text | naive | 0.9295 | 0.9294 | 35,918 | 8,980 |
| Title+text | leak_free | 0.9184 | 0.9177 | 31,060 | 7,766 |


### Top Features – body, naive mode

**Pushing toward FAKE:**

- `trump` (-3.9622)
- `obama` (-2.5971)
- `news` (-2.0735)
- `time` (-1.9167)
- `post` (-1.8987)
- `hillary` (-1.8303)
- `century` (-1.7880)
- `like` (-1.7745)
- `america` (-1.7595)
- `know` (-1.6924)

**Pushing toward REAL:**

- `reuters` (47.7450)
- `washington` (7.4150)
- `corrects` (5.5204)
- `paragraph` (3.9649)
- `story` (3.2566)
- `london` (3.1640)
- `york` (2.9830)
- `moscow` (2.9578)
- `berlin` (2.7170)
- `verified` (2.6853)

### Top Features – body, leak_free mode

**Pushing toward FAKE:**

- `trump` (-7.2299)
- `obama` (-6.0817)
- `gop` (-5.2184)
- `know` (-4.7167)
- `say` (-4.7014)
- `today` (-3.8559)
- `morning` (-3.7849)
- `night` (-3.7427)
- `cnn` (-3.6863)
- `america` (-3.5958)

**Pushing toward REAL:**

- `president` (7.5713)
- `said` (5.9999)
- `china` (4.8086)
- `britain` (4.6917)
- `administration` (4.5565)
- `representative` (4.4345)
- `thursday` (4.3719)
- `wednesday` (4.3011)
- `monday` (4.2961)
- `minister` (4.2336)

## Interpretation

- **Naive mode** achieves **99.7%** accuracy from the first 50 characters of the body.
- **Leak-free mode** drops to **90.0%**.
- The **9.7% gap** shows how much the naive model gets from the
  Reuters dateline alone.