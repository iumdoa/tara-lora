import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Thread, Event
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serve_writer import WriterServer, token_budget, validate_request, artifact_identity


class FakeBackend:
    info = {'id': 'tara-test', 'max_context_tokens': 100, 'max_output_tokens': 20}
    def __init__(self):
        self.cancelled = Event()
    def prepare(self, messages, requested, temperature):
        return messages[1]['content'], token_budget(len(messages[1]['content']), requested, 100, 20)
    def generate(self, prepared, emit, cancelled):
        text, limit = prepared
        if text == 'error':
            raise RuntimeError('generation failed')
        if text == 'wait':
            emit('等待。')
            cancelled.wait(5)
            if cancelled.is_set():
                self.cancelled.set()
            return 'aborted'
        emit('他推开门。')
        return 'length' if limit == 1 else 'stop'


class WriterServiceTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.server = WriterServer(('127.0.0.1', 0), self.backend, 'test-key')
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
    def request(self, text='写正文', **changes):
        body = {'model': 'tara-test', 'stream': True, 'max_tokens': 20,
                'messages': [{'role': 'system', 'content': '写作'}, {'role': 'user', 'content': text}], **changes}
        return Request(self.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                       headers={'Authorization': 'Bearer test-key', 'Content-Type': 'application/json'})
    def test_models_and_authentication(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + '/v1/models')
        self.assertEqual(error.exception.code, 401)
        req = Request(self.url + '/v1/models', headers={'Authorization': 'Bearer test-key'})
        with urlopen(req) as response:
            self.assertEqual(json.load(response)['data'][0]['id'], 'tara-test')
    def test_stream_and_finish_reasons(self):
        for limit, reason in [(20, 'stop'), (1, 'length')]:
            with urlopen(self.request(max_tokens=limit)) as response:
                stream = response.read().decode()
            self.assertIn('他推开门。', stream)
            self.assertIn(f'"finish_reason": "{reason}"', stream)
            self.assertTrue(stream.endswith('data: [DONE]\n\n'))
    def test_failure_is_not_success(self):
        with urlopen(self.request('error')) as response:
            stream = response.read().decode()
        self.assertIn('generation failed', stream)
        self.assertNotIn('[DONE]', stream)
    def test_invalid_model_tools_and_context_are_blocked(self):
        for text, changes, status in [('写正文', {'model': 'old-weights'}, 404),
                                      ('写正文', {'tools': [{}]}, 400), ('长' * 90, {}, 400)]:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.request(text, **changes))
            self.assertEqual(error.exception.code, status)
    def test_disconnect_cancels_generation_and_releases_slot(self):
        response = urlopen(self.request('wait'))
        response.readline()
        with self.assertRaises(HTTPError) as error:
            urlopen(self.request())
        self.assertEqual(error.exception.code, 429)
        response.close()
        self.assertTrue(self.backend.cancelled.wait(3))
        for _ in range(30):
            if not self.server.generation_lock.locked():
                break
            time.sleep(.01)
        self.assertFalse(self.server.generation_lock.locked())
    def test_no_prompt_truncation(self):
        self.assertEqual(token_budget(80, 30, 100, 20), 20)
        with self.assertRaises(ValueError):
            token_budget(81, 30, 100, 20)
    def test_identity_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as root:
            base, adapter = Path(root) / 'base', Path(root) / 'adapter'
            base.mkdir(); adapter.mkdir()
            (base / 'config.json').write_text('{}')
            (adapter / 'adapter_config.json').write_text('{}')
            with self.assertRaises(ValueError):
                artifact_identity(base, adapter, 4096, 1024)
            weight = adapter / 'adapter_model.safetensors'
            weight.write_bytes(b'v1')
            first = artifact_identity(base, adapter, 4096, 1024)
            weight.write_bytes(b'v2')
            self.assertNotEqual(first['id'], artifact_identity(base, adapter, 4096, 1024)['id'])


if __name__ == '__main__':
    unittest.main()
