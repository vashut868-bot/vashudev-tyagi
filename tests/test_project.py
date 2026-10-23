import bootstrap
import unittest
import json
import threading
import base64
import io
import urllib.request
import urllib.error
import numpy as np
from PIL import Image, ImageDraw
from ml import split_data, preprocess
from app import create_server

class ProjectTests(unittest.TestCase):
    def test_split_has_no_overlap(self):
        data, train, validation, test = split_data()
        self.assertEqual(len(set(train) | set(validation) | set(test)), len(data.target))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        for subset in (train, validation, test):
            self.assertEqual(set(data.target[subset]), set(range(10)))

    def test_blank_and_polarity(self):
        for color in ('white', 'black'):
            with self.assertRaises(ValueError):
                preprocess(Image.new('RGB', (100, 100), color))
        light = Image.new('L', (100, 100), 255)
        ImageDraw.Draw(light).line([(45, 15), (45, 85)], fill=0, width=12)
        dark = Image.fromarray(255 - np.asarray(light))
        np.testing.assert_allclose(preprocess(light), preprocess(dark))
        self.assertEqual(preprocess(light).shape, (1, 64))
        self.assertTrue(np.isfinite(preprocess(light)).all())

    def test_http_and_model(self):
        server = create_server(0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base + '/api/metrics') as response:
                metrics = json.load(response)
            self.assertGreater(metrics['test_accuracy'], .9)
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), metrics['test_samples'])
            with urllib.request.urlopen(base + '/api/sample?index=0') as response:
                sample = json.load(response)
            self.assertAlmostEqual(sum(row['score'] for row in sample['scores']), 1)
            self.assertEqual(len(sample['pixels']), 8)
            drawing = Image.new('L', (100, 100), 255)
            ImageDraw.Draw(drawing).line([(50, 15), (50, 85)], fill=0, width=12)
            buffer = io.BytesIO()
            drawing.save(buffer, format='PNG')
            body = json.dumps({'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}).encode()
            request = urllib.request.Request(base + '/api/predict', data=body, method='POST')
            with urllib.request.urlopen(request) as response:
                prediction = json.load(response)
            self.assertIn(prediction['digit'], range(10))
            request = urllib.request.Request(base + '/api/predict', data=b'{"image":"invalid"}', method='POST')
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
            with urllib.request.urlopen(base) as response:
                self.assertIn(b'DigitLab', response.read())
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

if __name__ == '__main__':
    unittest.main()
