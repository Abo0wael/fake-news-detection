# Model Comparison: Naive vs. Leak-Free

| Mode | Model | CV F1 (mean±std) | Accuracy | Precision | Recall | F1 (macro) |
|------|-------|------------------|----------|-----------|--------|------------|
| naive | LogisticRegression | 0.9924±0.0009 | 0.9921 | 0.9920 | 0.9922 | 0.9921 |
| naive | LinearSVC | 0.9972±0.0003 | 0.9974 | 0.9974 | 0.9974 | 0.9974 |
| naive | MultinomialNB | 0.9647±0.0019 | 0.9661 | 0.9661 | 0.9660 | 0.9661 |
| leak_free | LogisticRegression | 0.9804±0.0014 | 0.9790 | 0.9794 | 0.9784 | 0.9789 |
| leak_free | LinearSVC | 0.9878±0.0008 | 0.9884 | 0.9886 | 0.9881 | 0.9883 |
| leak_free | MultinomialNB | 0.9519±0.0030 | 0.9524 | 0.9518 | 0.9524 | 0.9521 |

## Per-Class Metrics (Leak-Free)

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| LogisticRegression | Fake | 0.9833 | 0.9709 | 0.9771 |
| LogisticRegression | Real | 0.9754 | 0.9859 | 0.9806 |
| LinearSVC | Fake | 0.9907 | 0.9841 | 0.9874 |
| LinearSVC | Real | 0.9865 | 0.9921 | 0.9893 |
| MultinomialNB | Fake | 0.9441 | 0.9531 | 0.9486 |
| MultinomialNB | Real | 0.9595 | 0.9517 | 0.9556 |