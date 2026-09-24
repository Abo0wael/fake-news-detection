# Style Stress Test (extra analysis)

Logistic Regression on leak_free text, with and without common Reuters style markers.
This is **not** the main result; it measures how much accuracy depends on house style.

Removed tokens: `aug`, `dec`, `feb`, `friday`, `jan`, `monday`, `nov`, `oct`, `reporter`, `said`, `saturday`, `say`, `saying`, `sept`, `spokesman`, `spokesperson`, `spokeswoman`, `statement`, `sunday`, `tell`, `thursday`, `told`, `tuesday`, `wednesday`

| Variant | Accuracy | F1 (macro) |
|---------|----------|------------|
| leak_free (main) | 0.9790 | 0.9789 |
| leak_free + style markers removed | 0.9751 | 0.9750 |

## Top features after removing style markers

| Rank | → FAKE | → REAL |
|------|--------|--------|
| 1 | `gop` (-7.59) | `president donald` (+8.64) |
| 2 | `watch` (-7.34) | `representative` (+4.62) |
| 3 | `even` (-6.11) | `minister` (+4.55) |
| 4 | `president trump` (-5.93) | `president barack` (+4.38) |
| 5 | `obama` (-5.82) | `presidential` (+4.35) |
| 6 | `america` (-5.69) | `democratic` (+4.02) |
| 7 | `hillary` (-5.23) | `comment` (+3.72) |
| 8 | `president obama` (-5.05) | `president` (+3.64) |
| 9 | `like` (-4.99) | `rival` (+3.62) |
| 10 | `breaking` (-4.74) | `republican presidential` (+3.38) |