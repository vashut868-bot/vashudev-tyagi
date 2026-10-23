"""Local-only web demo. Run: python app.py"""
import bootstrap
import base64
import io
import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from ml import ARTIFACTS, split_data, train_models, preprocess

class Handler(BaseHTTPRequestHandler):
    def send(self, status, value, content_type='application/json'):
        body = json.dumps(value).encode() if content_type == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def result(self, features):
        probabilities = self.server.model.predict_proba(features)[0]
        return {'digit': int(self.server.model.predict(features)[0]),
                'scores': [{'digit': int(label), 'score': float(score)} for label, score in zip(self.server.model.classes_, probabilities)],
                'pixels': features.reshape(8, 8).tolist()}

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            self.send(200, (bootstrap.ROOT / 'web' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif url.path == '/api/metrics':
            self.send(200, self.server.metrics)
        elif url.path == '/api/sample':
            try:
                index = int(parse_qs(url.query).get('index', ['0'])[0])
                if not 0 <= index < len(self.server.test):
                    raise ValueError()
                sample = self.server.test[index]
                result = self.result(self.server.data.data[sample:sample+1])
                result['actual'] = int(self.server.data.target[sample])
                self.send(200, result)
            except ValueError:
                self.send(400, {'error': 'Invalid test sample index.'})
        else:
            self.send(404, {'error': 'Not found.'})

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send(404, {'error': 'Not found.'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 6_000_000:
                raise ValueError('Image must be smaller than 4 MB.')
            payload = json.loads(self.rfile.read(length))
            encoded = payload['image'].split(',', 1)[1]
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError('Use an image smaller than 16 megapixels.')
                features = preprocess(image)
            self.send(200, self.result(features))
        except (ValueError, KeyError, TypeError, IndexError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            self.send(400, {'error': str(exc) or 'Invalid image.'})

def create_server(port):
    if not (ARTIFACTS / 'model.joblib').exists() or not (ARTIFACTS / 'metrics.json').exists():
        model, metrics = train_models()
    else:
        model = joblib.load(ARTIFACTS / 'model.joblib')
        metrics = json.loads((ARTIFACTS / 'metrics.json').read_text())
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.model, server.metrics = model, metrics
    server.data, _, _, server.test = split_data()
    return server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f'DigitLab is running at http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
