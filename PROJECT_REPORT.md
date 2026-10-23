# Handwritten Digit Recognition Using Machine Learning

**Student:** [vashudevtyagi]  
**Roll number:** [49]  
**Department / institution:** [B.tech/AIML]  
**Project guide:** [AIML]  
**Academic year:** [2026-27]

## Abstract

This project implements a local application that classifies isolated handwritten digits from zero to nine. It compares a support vector machine, a k-nearest neighbors classifier and a random forest using the scikit-learn digits dataset. Model selection uses an independent validation partition, followed by evaluation on an untouched test partition. A browser interface supports drawing, image upload, probability inspection and confusion-matrix analysis. The project demonstrates a complete classical machine learning workflow while distinguishing benchmark performance from performance on user-created images.

## 1. Problem definition and objectives

The task is supervised multiclass classification: map a grayscale image of one digit to a label in {0, …, 9}. Objectives are to construct a reproducible pipeline, compare algorithms fairly, calculate useful metrics, expose predictions through a local application and explain failure cases. Reading full documents or recognizing multiple characters is outside the implemented scope.

## 2. Dataset

The bundled scikit-learn digits dataset contains 1,797 labeled 8×8 grayscale images. Each image is flattened into 64 features with intensity values from 0 to 16. It is derived from the UCI optical handwritten digits dataset; it is not MNIST. It is small enough for a CPU-based educational experiment. Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

## 3. System architecture

Browser drawing/upload → HTTP request → image preprocessing → trained classifier → digit and probability estimates → results dashboard.

Training proceeds separately: bundled data → stratified train/validation/test split → three candidate pipelines → validation macro F1 comparison → refit winner on development data → final test evaluation → saved model and JSON metrics.

The HTTP server binds to localhost. The client uses standard browser APIs, with no frontend build step. Images are processed in memory. The application is a local demonstration server rather than a public hosting service.

## 4. Experimental methodology

The data is split into 1,077 training, 360 validation and 360 test examples using stratified sampling with seed 42. Scaling is fitted inside the SVM and KNN pipelines, preventing validation/test information from influencing training statistics. Random forest uses raw intensities.

The SVM uses an RBF kernel with C=10 and gamma='scale'. KNN uses three neighbors. Random forest uses 200 trees. These are fixed baseline configurations, not an exhaustive hyperparameter search. The highest validation macro F1 selects the winner. That model is refitted on 1,437 training-plus-validation examples and evaluated once on the 360 held-out test examples. Test accuracy must not guide subsequent manual parameter changes; further experimentation would require a new evaluation protocol.

Image inputs are converted to grayscale with transparency composited onto white. Border brightness determines whether to invert the image. Background subtraction, foreground cropping, aspect-ratio-preserving resizing and centering produce an 8×8 bright-on-dark image scaled to [0,16]. This heuristic can fail for textured backgrounds, unusual framing or writing styles.

## 5. Evaluation

Accuracy is the fraction of correct predictions. Per-class precision measures the fraction of predicted members of a class that are correct. Recall measures the fraction of true class members recovered. F1 is the harmonic mean of precision and recall; macro F1 gives each digit equal weight. The confusion matrix records counts of actual-versus-predicted labels and identifies commonly confused digits.

Actual measured results are saved in `artifacts/metrics.json` and summarized in `RESULTS.md`. The dashboard reads those same results. Fit time depends on hardware and environment. No invented accuracy values are used.

## 6. Verification

Automated checks verify that the partitions do not overlap, every partition contains all ten classes, blank images are rejected, equivalent light/dark drawings preprocess consistently, probability estimates sum to one, malformed input returns a client error, and the application serves the interface and test results. A minimum benchmark accuracy check catches major pipeline failures; it does not establish real-world reliability.

## 7. Limitations and responsible interpretation

The small low-resolution dataset does not represent all handwriting styles, phone cameras or real documents. The classifier always chooses one of ten digits and has no reliable unknown-input detector. Probability estimates are not calibrated guarantees. The experiment uses one deterministic split rather than repeated trials, and image preprocessing has not been validated on a separately collected handwritten dataset. Benchmark results therefore support only performance on this test partition. This work is an educational integration of established methods, not a claim of algorithmic novelty.

## 8. Future work

Collect a consented, labeled sample of actual user drawings and evaluate it separately. Compare a CNN against the classical baselines on a larger dataset. Study deskewing, augmentation and calibration. Add explicit rejection of blank, ambiguous and non-digit inputs using an independently evaluated threshold or classifier. For robust research claims, use cross-validation for selection and preserve a final test dataset.

## 9. Conclusion

The implemented system connects model training, fair selection, evaluation and interactive inference in one reproducible project. Its main educational value is demonstrating the full workflow and making errors visible. Further data collection and evaluation are required before using it for operational document recognition.

## References

1. scikit-learn, load_digits documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
2. UCI Machine Learning Repository, Optical Recognition of Handwritten Digits: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
3. scikit-learn, Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
