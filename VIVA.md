# Viva preparation and demo outline

## Five-minute demonstration

1. **Problem (30 seconds):** classify one handwritten digit into ten possible classes.
2. **Data (45 seconds):** explain 1,797 images, 64 pixels/features and labels 0–9.
3. **Experiment (60 seconds):** explain stratification, the three partitions, candidate models and validation-based selection.
4. **Demo (90 seconds):** show a test sample, draw a digit, upload an image, inspect pixels and explain any incorrect prediction honestly.
5. **Results and limits (75 seconds):** show actual metrics and the confusion matrix; distinguish benchmark data from user drawings.

## Questions and suggested answers

**What type of machine learning is this?** Supervised multiclass classification because training examples include correct digit labels.

**Is this deep learning?** No. The implemented classifiers are classical ML methods. A CNN is a possible extension.

**Why three data partitions?** Training fits parameters, validation selects the model, and test estimates performance after selection.

**What is data leakage?** Information from evaluation data influencing training or selection. Here scaling is fitted in each training pipeline and the final test set is excluded from selection.

**Why stratification?** It approximately preserves digit proportions in each partition.

**Why scale features?** Distance-based KNN and margin-based SVM are sensitive to feature scales. A pipeline ensures scaling is learned from training data only.

**How does SVM work?** It learns separating decision boundaries; an RBF kernel enables nonlinear class separation.

**How does KNN work?** It predicts using the labels of nearby training examples in feature space.

**How does random forest work?** It combines many decision trees trained with randomization and aggregates their predictions.

**What is macro F1?** Compute F1 for each digit, then average all ten equally.

**Why can a high-accuracy model misread my drawing?** Mouse strokes, cropping and backgrounds differ from the training distribution. Benchmark accuracy does not measure those conditions.

**Are probabilities guaranteed confidence?** No. They are model estimates and have not been independently calibrated. SVC's predicted class can differ from the maximum estimated probability.

**What happens with a letter or two digits?** The classifier still chooses a digit. Reliable rejection and character segmentation are not implemented.

**What makes the experiment reproducible?** Fixed split seeds, fixed model settings, saved metrics and pinned direct dependencies. Timing and numerical details can still vary across environments.

**What is your project's contribution?** Integrating preprocessing, a fair baseline comparison, local inference, interactive visualization and documented limitations. Do not claim a new algorithm.

**How could it become a major project?** Add a CNN comparison, an independently collected test set, ablation experiments, robustness analysis and validated uncertainty/rejection handling.
