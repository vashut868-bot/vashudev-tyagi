"""Reproducible model selection, held-out evaluation, and image preprocessing."""
import bootstrap
import json
import time
import numpy as np
import joblib
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ARTIFACTS = bootstrap.ROOT / 'artifacts'

def split_data():
    data = load_digits()
    train, test = train_test_split(np.arange(len(data.target)), test_size=0.2,
                                  stratify=data.target, random_state=42)
    train, validation = train_test_split(train, test_size=0.25,
                                         stratify=data.target[train], random_state=42)
    return data, train, validation, test

def train_models():
    data, train, validation, test = split_data()
    candidates = {
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', probability=True, random_state=42)),
        'K-Nearest Neighbors': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
    }
    rows = []
    for name, model in candidates.items():
        start = time.perf_counter()
        model.fit(data.data[train], data.target[train])
        predictions = model.predict(data.data[validation])
        rows.append({'model': name, 'validation_accuracy': accuracy_score(data.target[validation], predictions),
                     'validation_macro_f1': f1_score(data.target[validation], predictions, average='macro'),
                     'training_seconds': round(time.perf_counter() - start, 3)})
    winner = max(rows, key=lambda row: row['validation_macro_f1'])['model']
    model = candidates[winner]
    development = np.concatenate([train, validation])
    model.fit(data.data[development], data.target[development])
    predicted = model.predict(data.data[test])
    report = {'selected_model': winner, 'seed': 42, 'total_samples': len(data.target),
              'training_samples': len(train), 'validation_samples': len(validation), 'test_samples': len(test),
              'comparison': rows, 'test_accuracy': accuracy_score(data.target[test], predicted),
              'test_macro_f1': f1_score(data.target[test], predicted, average='macro'),
              'classification_report': classification_report(data.target[test], predicted, output_dict=True),
              'confusion_matrix': confusion_matrix(data.target[test], predicted, labels=list(range(10))).tolist()}
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / 'model.joblib')
    (ARTIFACTS / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return model, report

def preprocess(image):
    """Convert a single digit to bright foreground, crop, resize and center."""
    image = ImageOps.exif_transpose(image).convert('RGBA')
    background = Image.new('RGBA', image.size, 'white')
    background.alpha_composite(image)
    gray = np.asarray(background.convert('L'), dtype=np.float32)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.median(border) > 127:
        gray = 255 - gray
    gray = np.clip(gray - np.percentile(gray, 10), 0, 255)
    if gray.max() < 25:
        raise ValueError('Draw or upload a visible digit first.')
    ys, xs = np.where(gray > max(25, gray.max() * 0.2))
    if len(xs) < 3:
        raise ValueError('The image contains too little detail. Try a larger digit.')
    crop = Image.fromarray(gray[ys.min():ys.max()+1, xs.min():xs.max()+1].astype('uint8'))
    # The digits dataset largely fills its vertical extent with modest horizontal margins.
    scale = min(6 / crop.width, 8 / crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (8, 8), 0)
    canvas.paste(crop, ((8-crop.width)//2, (8-crop.height)//2))
    return np.asarray(canvas, dtype=np.float64).reshape(1, 64) / 255 * 16

if __name__ == '__main__':
    _, metrics = train_models()
    print(json.dumps(metrics, indent=2))
