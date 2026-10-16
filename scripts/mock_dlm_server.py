#!/usr/bin/env python3
"""Loopback-only oracle candidate server for vLLM integration tests, NOT a DLM."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from draft_test_support import PROTOCOL, prefix_digest, tokenizer_digest

MODES = ('all_accept', 'first_reject', 'middle_reject', 'no_draft', 'stale', 'invalid_token', 'timeout')


def make_handler(traces, tokenizer_sha256, log_path, *, delay=0.6):
    state = {'mode': 'no_draft'}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self.send(200 if self.path == '/health' else 404, {'type': 'synthetic-oracle'})

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Request size')
                obj = json.loads(self.rfile.read(size))
                if self.path == '/control':
                    if obj['mode'] not in MODES:
                        raise ValueError('Unknown test mode')
                    with lock:
                        state['mode'] = obj['mode']
                        state['draft_override'] = obj.get('draft_override')
                        if 'traces' in obj:
                            if not isinstance(obj['traces'], list):
                                raise ValueError('Trace list required')
                            traces[:] = obj['traces']
                    self.send(200, {'mode': obj['mode']})
                    return
                if self.path != '/v1/draft':
                    self.send(404, {'error': 'Unknown endpoint'})
                    return
                if (obj['protocol'] != PROTOCOL or obj['draft_mode'] != 'greedy'
                        or obj['tokenizer_sha256'] != tokenizer_sha256
                        or obj['prefix_sha256'] != prefix_digest(obj['prefix_token_ids'])):
                    raise ValueError('Protocol/tokenizer/prefix mismatch')
                prefix, k = obj['prefix_token_ids'], obj['max_draft_tokens']
                if type(k) is not int or not 0 <= k <= 32:
                    raise ValueError('Invalid draft count')
                with lock:
                    mode = state['mode']
                draft = []
                matched = False
                for trace in traces:
                    full = trace['prompt_token_ids'] + trace['output_token_ids']
                    if len(prefix) >= len(trace['prompt_token_ids']) and full[:len(prefix)] == prefix:
                        draft = full[len(prefix):len(prefix) + k]
                        matched = True
                        break
                if mode == 'no_draft':
                    draft = []
                elif draft and mode in ('first_reject', 'middle_reject'):
                    at = 0 if mode == 'first_reject' else len(draft) // 2
                    draft[at] = (draft[at] + 1) % 50272
                elif mode == 'invalid_token':
                    draft = [-1]
                elif mode == 'timeout':
                    time.sleep(delay)
                with lock:
                    if state.get('draft_override') is not None:
                        draft = state['draft_override']
                reply = {key:obj[key] for key in ('protocol', 'request_id', 'prefix_sha256',
                                                  'tokenizer_sha256', 'draft_mode')}
                reply['draft_token_ids'] = draft
                if mode == 'stale':
                    reply['request_id'] = 'stale-response'
                record = {'mode': mode, 'prefix_tokens': len(prefix), 'matched_oracle': matched,
                          'request_id': obj['request_id'], 'prefix_sha256': obj['prefix_sha256'],
                          'draft_token_ids': draft}
                if log_path:
                    with lock, Path(log_path).open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                self.send(200, reply)
            except (ValueError, KeyError, TypeError):
                self.send(400, {'error': 'Invalid test request'})

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, help='Optional oracle fixtures; validator can populate them')
    p.add_argument('--model', type=Path, default=Path('.cache/models/facebook/opt-6.7b'))
    p.add_argument('--port', type=int, default=8012)
    p.add_argument('--log', type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    data = (json.loads(args.traces.read_text()) if args.traces else
            {'traces': [], 'tokenizer_sha256': tokenizer_digest(args.model)})
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
        make_handler(data['traces'], data['tokenizer_sha256'], args.log))
    print(f'Synthetic oracle DLM listening on 127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
