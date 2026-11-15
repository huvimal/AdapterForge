import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from api import create_app


class ApiTests(unittest.TestCase):
    def test_injected_test_engine(self):
        with TestClient(create_app(engine=lambda text: {'raw_output': text, 'validation': {}})) as client:
            self.assertTrue(client.get('/health').json()['ready'])
            self.assertEqual(client.post('/extract', json={'text': 'Demo'}).json()['raw_output'], 'Demo')

    def test_input_limits(self):
        with TestClient(create_app(engine=lambda text: {})) as client:
            for value in ('', '   ', 'a' * 1201):
                self.assertEqual(client.post('/extract', json={'text': value}).status_code, 422)
            self.assertEqual(client.post('/extract', json={'text': 'Demo', 'adapter': 'remote'}).status_code, 422)

    def test_busy_request_rejected(self):
        entered, release = threading.Event(), threading.Event()
        def engine(text):
            entered.set()
            release.wait(timeout=5)
            return {'ok': True}
        with TestClient(create_app(engine=engine)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(client.post, '/extract', json={'text': 'first'})
                self.assertTrue(entered.wait(timeout=5))
                try:
                    self.assertEqual(client.post('/extract', json={'text': 'second'}).status_code, 503)
                finally:
                    release.set()
                self.assertEqual(first.result().status_code, 200)
