# DigitLab — Handwritten Digit Recognition

A B.Tech AI/ML mini-project with a browser demo, reproducible training, three-model comparison, and held-out evaluation. Python handles inference; HTML, CSS and JavaScript provide a drawing canvas and results dashboard. No cloud account, API key, or dataset download is required after installing dependencies.

## Run on this computer

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Then open http://127.0.0.1:8000 in a browser. Leave the terminal open; press Ctrl+C to stop. This project includes dependencies in `.packages` for the bundled Python available on this computer.

## Set up on another computer

Install Python 3.12, then run from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ml.py
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. If copying the project, exclude `.packages` and `.venv`; install requirements on the destination. Retrain to produce model artifacts compatible with that environment. Only load model artifacts you trust, because joblib uses Python serialization.

## Demonstration

1. Click **Try test sample** to inspect an unseen dataset digit and its true label.
2. Clear the canvas, draw one large digit, and click **Recognize digit**.
3. Upload a tightly framed PNG/JPEG/WebP containing one digit on a plain background.
4. Compare the processed 8×8 input with your original drawing.
5. Inspect validation results and the test confusion matrix below the demo.

The sample button predicts original dataset pixels. Drawing/upload prediction uses preprocessing and is a separate, harder input distribution. Benchmark accuracy must not be described as real-world drawing accuracy. Scores are uncalibrated probability estimates; the selected digit follows the classifier's prediction, which can differ from the highest probability estimate for SVC.

## Files

| File | Purpose |
| --- | --- |
| `ml.py` | Data splits, model comparison, final training, preprocessing |
| `app.py` | Local HTTP server and prediction endpoints |
| `web/index.html` | Responsive drawing/upload interface and dashboard |
| `artifacts/model.joblib` | Trained model including preprocessing pipeline |
| `artifacts/metrics.json` | Actual reproducible evaluation results |
| `PROJECT_REPORT.md` | Academic report and methodology |
| `VIVA.md` | Demonstration outline and viva preparation |
| `tests/test_project.py` | Split integrity, preprocessing and API tests |

## Train and verify

```powershell
python ml.py
python -m unittest discover -s tests -v
python app.py --port 8001
```

Use your working Python executable (the virtual environment or bundled runtime) for these commands. The first app launch trains if artifacts are missing. Training explicitly overwrites generated model/metrics files. Model selection uses validation macro F1, never test scores; the winner is refitted on training plus validation. No external service receives uploaded images, and the application does not save them. The server binds only to 127.0.0.1 and is for a local academic demonstration.

## Dataset and references

- [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

This is a classical ML mini-project, not a novel research contribution. For a major/final-year project, extend it with a larger dataset, CNN baseline, a separately collected handwriting test set, and a documented robustness study. Add your name, college, guide and department to the report before submission.
