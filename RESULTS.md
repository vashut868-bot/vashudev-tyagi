# Measured experiment results

Seed: 42. Training: 1,077 examples; validation: 360; final test: 360.

| Model | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: |
| Support Vector Machine | 99.17% | 0.9916 |
| K-Nearest Neighbors | 96.39% | 0.9640 |
| Random Forest | 98.06% | 0.9806 |

The SVM was selected using validation macro F1 and refitted on the combined 1,437 development examples.

- Final test accuracy: **98.06% (353 / 360)**.
- Final test macro F1: **0.9805**.
- Seven errors: 1→4, 4→7, 7→5, 8→1, 8→4, 9→6, and 9→7 (one example each).

These values describe this held-out dataset partition, not accuracy on drawings or uploaded photographs. Detailed per-class results and the full confusion matrix are in `artifacts/metrics.json`. If you change the experiment and retrain, update this summary from the new JSON results.
