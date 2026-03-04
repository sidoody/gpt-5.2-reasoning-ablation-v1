# Final Statistical Report

All outputs are derived deterministically from files under `results/` and `scores/`.

## Per-variant diagnosis accuracy

| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg reasoning tokens | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| none | 897 | 0.639 | [0.607, 0.670] | 613.61 | 0.00 | 2.608 |
| low | 897 | 0.664 | [0.633, 0.695] | 782.13 | 163.83 | 5.549 |
| medium | 897 | 0.673 | [0.642, 0.703] | 935.39 | 315.06 | 10.807 |
| high | 897 | 0.688 | [0.657, 0.717] | 1088.05 | 468.38 | 13.567 |

## All-pairs exact McNemar tests

| Comparison | N | Accuracy A | Accuracy B | |Delta| | A-only correct | B-only correct | Discordant total | Exact p-value | Holm-adjusted p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none_vs_low | 897 | 0.639 | 0.664 | 0.026 | 48 | 71 | 119 | 0.04326826 | 0.15005693 |
| none_vs_medium | 897 | 0.639 | 0.673 | 0.035 | 44 | 75 | 119 | 0.00572686 | 0.02863431 |
| none_vs_high | 897 | 0.639 | 0.688 | 0.049 | 44 | 88 | 132 | 0.00016024 | 0.00096141 |
| low_vs_medium | 897 | 0.664 | 0.673 | 0.009 | 41 | 49 | 90 | 0.46079247 | 0.46079247 |
| low_vs_high | 897 | 0.664 | 0.688 | 0.023 | 36 | 57 | 93 | 0.03751423 | 0.15005693 |
| medium_vs_high | 897 | 0.673 | 0.688 | 0.014 | 36 | 49 | 85 | 0.19276044 | 0.38552087 |

## Efficiency frontier

| Variant | Accuracy | Avg total tokens | Avg reasoning tokens | Avg latency (s) |
|---|---:|---:|---:|---:|
| none | 0.639 | 613.61 | 0.00 | 2.608 |
| low | 0.664 | 782.13 | 163.83 | 5.549 |
| medium | 0.673 | 935.39 | 315.06 | 10.807 |
| high | 0.688 | 1088.05 | 468.38 | 13.567 |
