"""Local OpenAI-compatible TARA Writer. Load once; serialize generation; never truncate input."""
import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Lock, Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_request(body, model_id):
    if not isinstance(body, dict) or body.get('model') != model_id:
        raise RequestError('Unknown model; use the exact versioned id from /v1/models.', 404)
    if body.get('stream') is not True:
        raise RequestError('This Writer requires stream=true.')
    if body.get('tools') or body.get('stop') or body.get('n', 1) != 1:
        raise RequestError('Tools, stop sequences and multiple completions are unsupported.')
    messages = body.get('messages')
    if not isinstance(messages, list) or [m.get('role') if isinstance(m, dict) else None for m in messages] != ['system', 'user']:
        raise RequestError('Writer requires exactly one system and one user message.')
    if any(not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages):
        raise RequestError('Messages must contain nonempty text.')
    maximum = body.get('max_tokens', 1024)
    temperature = body.get('temperature', 0.7)
    if type(maximum) is not int or maximum < 1:
        raise RequestError('max_tokens must be a positive integer.')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise RequestError('temperature must be between 0 and 2.')
    return messages, maximum, temperature


def token_budget(prompt_tokens, requested, context_limit, output_limit):
    # Reserve the requested output up to the configured service cap; never clip the prompt.
    output = min(requested, output_limit)
    if prompt_tokens + output > context_limit:
        raise RequestError(f'Context budget exceeded: prompt={prompt_tokens}, output={output}, limit={context_limit}. Shorten input or explicitly reduce max_tokens.')
    return output


def artifact_identity(base, adapter, context_limit, output_limit):
    files = [('base', p) for p in sorted(base.iterdir()) if p.suffix in ('.json', '.safetensors', '.bin', '.model')]
    files += [('adapter', p) for p in sorted(adapter.iterdir()) if p.suffix in ('.json', '.safetensors', '.bin', '.model')]
    if not (adapter / 'adapter_config.json').is_file() or not any(p.name.startswith('adapter_model.') for _, p in files):
        raise ValueError('Adapter config and weights are required; no silent base-model fallback.')
    digest = hashlib.sha256()
    for label, path in files:
        digest.update(f'{label}/{path.name}\0'.encode())
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
    prompt_hash = hashlib.sha256(Path(__file__).with_name('prompt_builder.py').read_bytes()).hexdigest()
    service_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest.update(f'{prompt_hash}:{service_hash}:{context_limit}:{output_limit}'.encode())
    revision = digest.hexdigest()
    return {'id': f'tara-{revision}', 'object': 'model', 'owned_by': 'local-lora',
            'revision': revision, 'protocol': 'tara5_action_v1', 'prompt_sha256': prompt_hash,
            'max_context_tokens': context_limit, 'max_output_tokens': output_limit}


class TransformersBackend:
    def __init__(self, base, adapter, context_limit, output_limit):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self.info = artifact_identity(base, adapter, context_limit, output_limit)
        tokenizer_path = adapter if (adapter / 'tokenizer_config.json').exists() else base
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None)
        self.model = PeftModel.from_pretrained(model, adapter, local_files_only=True).eval()
        model_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if model_limit and context_limit > model_limit:
            raise ValueError(f'Configured context exceeds base model limit {model_limit}.')

    def prepare(self, messages, requested, temperature):
        # Same template and special-token policy as train_lora_local.py.
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        count = inputs['input_ids'].shape[1]
        output = token_budget(count, requested, self.info['max_context_tokens'], self.info['max_output_tokens'])
        return inputs, output, temperature

    def generate(self, prepared, emit, cancelled):
        import torch
        from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text and not cancelled.is_set():
                    emit(text)
        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()
        inputs, maximum, temperature = prepared
        inputs = inputs.to(self.model.device)
        streamer = Streamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {'temperature': temperature, 'top_p': 0.9} if temperature > 0 else {}
        with torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=maximum, do_sample=temperature > 0,
                streamer=streamer, stopping_criteria=StoppingCriteriaList([Cancel()]),
                pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        if cancelled.is_set():
            return 'aborted'
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return 'stop' if result[0, -1].item() in eos else 'length'


class WriterServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, backend, api_key=''):
        self.backend, self.api_key, self.generation_lock = backend, api_key, Lock()
        super().__init__(address, WriterHandler)


class WriterHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Never log prompts or authorization headers.

    def authorized(self):
        key = self.server.api_key
        if key and not hmac.compare_digest(self.headers.get('Authorization', ''), f'Bearer {key}'):
            self.json_response(401, {'error': {'message': 'Invalid API key.'}})
            return False
        return True

    def json_response(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == '/v1/models':
            self.json_response(200, {'object': 'list', 'data': [self.server.backend.info]})
        else:
            self.json_response(404, {'error': {'message': 'Unknown endpoint.'}})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != '/v1/chat/completions':
            self.json_response(404, {'error': {'message': 'Unknown endpoint.'}})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2 * 1024 * 1024:
                raise RequestError('Invalid request size.', 413)
            body = json.loads(self.rfile.read(length))
            values = validate_request(body, self.server.backend.info['id'])
        except (ValueError, UnicodeError) as error:
            self.json_response(getattr(error, 'status', 400), {'error': {'message': str(error)}})
            return
        if not self.server.generation_lock.acquire(blocking=False):
            self.json_response(429, {'error': {'message': 'Writer busy; retry after the active generation finishes.'}})
            return
        try:
            prepared = self.server.backend.prepare(*values)
        except Exception as error:
            self.server.generation_lock.release()
            self.json_response(getattr(error, 'status', 500), {'error': {'message': str(error)}})
            return
        events, cancelled = Queue(), Event()
        def run():
            try:
                reason = self.server.backend.generate(prepared, lambda text: events.put(('text', text)), cancelled)
                events.put(('done', reason))
            except Exception as error:
                events.put(('error', str(error)))
            finally:
                self.server.generation_lock.release()
        worker = Thread(target=run, daemon=True)
        worker.start()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            while True:
                try:
                    kind, value = events.get(timeout=0.25)
                except Empty:
                    self.wfile.write(b': heartbeat\n\n')
                    self.wfile.flush()
                    continue
                if kind == 'error':
                    self.sse({'error': {'message': value}})
                    break
                self.sse({'object': 'chat.completion.chunk', 'model': self.server.backend.info['id'],
                          'choices': [{'index': 0, 'delta': {'content': value} if kind == 'text' else {},
                                       'finish_reason': value if kind == 'done' else None}]})
                if kind == 'done':
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            cancelled.set()
            # Worker owns the generation lock until the GPU operation actually finishes.

    def sse(self, payload):
        self.wfile.write(('data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n').encode())
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--context-tokens', type=int, default=4096)
    parser.add_argument('--output-tokens', type=int, default=1024)
    args = parser.parse_args()
    key = os.environ.get('TARA_API_KEY', '')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not key:
        parser.error('Set TARA_API_KEY before binding a non-loopback address.')
    if not 0 < args.output_tokens < args.context_tokens:
        parser.error('Require 0 < output-tokens < context-tokens.')
    backend = TransformersBackend(args.base_model.resolve(), args.adapter.resolve(), args.context_tokens, args.output_tokens)
    server = WriterServer((args.host, args.port), backend, key)
    print(json.dumps({'endpoint': f'http://{args.host}:{server.server_port}/v1', **backend.info}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
