"""Exercise the compiled host tools against actual bounded HTTP/SSE fixtures."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from draft_test_support import tokenizer_digest
from mock_dlm_server import make_handler

BINARY = ROOT / 'build/draft/specferry'


class DraftToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.servers = []
        self.threads = []
        (self.root/'vocab.json').write_text('{}')
        (self.root/'merges.txt').write_text('#version: 0.2\n')
        (self.root/'prefix.json').write_text('[2,10,11]')
        (self.root/'prompt.txt').write_text('A normal user question')

    def tearDown(self):
        for server, thread in zip(self.servers, self.threads):
            server.shutdown()
            server.server_close()
            thread.join()
        self.tmp.cleanup()

    def server(self, handler):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
        thread.start()
        self.servers.append(server)
        self.threads.append(thread)
        return f'http://127.0.0.1:{server.server_port}'

    def run_tool(self, *args):
        return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=10)

    def test_tokenizer_fingerprint_matches_independent_fixture(self):
        result = self.run_tool('fingerprint', '--model-dir', str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), tokenizer_digest(self.root))

    def test_probe_distinguishes_empty_valid_reply_from_protocol_failure(self):
        handler = make_handler([], tokenizer_digest(self.root), None)
        url = self.server(handler)
        args = ('probe', '--model-dir', str(self.root), '--dlm-url', url+'/v1/draft',
                '--prefix-file', str(self.root/'prefix.json'))
        # An empty reply is valid but does not establish useful device drafting.
        empty = self.run_tool(*args)
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(json.loads(empty.stdout)['draft_token_ids'], [])
        from urllib.request import Request, urlopen
        request = Request(url+'/control', data=b'{"mode":"stale"}',
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=2):
            pass
        stale = self.run_tool(*args)
        self.assertNotEqual(stale.returncode, 0)
        self.assertEqual(json.loads(stale.stdout)['status'], 'fallback')

    def stream(self, kind):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if payload['temperature'] != 0 or not payload['stream']:
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                events = [
                    {'choices':[{'text':'你好', 'token_ids':[10,11], 'finish_reason':None}]},
                    {'choices':[{'text':'!', 'token_ids':[12], 'finish_reason':'length'}]},
                    {'choices':[], 'usage':{'prompt_tokens':4, 'completion_tokens':3, 'total_tokens':7}},
                ]
                if kind == 'error':
                    events = [{'error':{'message':'device-independent service error'}}]
                elif kind == 'wrong_count':
                    events[-1]['usage']['completion_tokens'] = 4
                body = ''.join('data: '+json.dumps(e,ensure_ascii=False)+'\r\n\r\n' for e in events)
                if kind != 'truncated':
                    body += 'data: [DONE]\r\n\r\n'
                # Split UTF-8 and SSE framing across arbitrarily small writes.
                try:
                    for byte in body.encode():
                        self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return self.server(Handler)

    def test_stream_outputs_utf8_and_records_actual_token_counts(self):
        directory = self.root/'success'
        result = self.run_tool('generate', '--base-url', self.stream('ok'),
                '--prompt-file', str(self.root/'prompt.txt'), '--output-dir', str(directory))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '你好!')
        self.assertEqual((directory/'output.txt').read_text(), '你好!')
        report = json.loads((directory/'report.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['token_ids'], [10,11,12])
        self.assertEqual(report['output_tokens'], 3)

    def test_truncated_error_and_wrong_count_streams_are_failures(self):
        for kind in ('truncated', 'error', 'wrong_count'):
            with self.subTest(kind=kind):
                directory = self.root/kind
                result = self.run_tool('generate', '--base-url', self.stream(kind),
                        '--prompt-file', str(self.root/'prompt.txt'), '--output-dir', str(directory))
                self.assertNotEqual(result.returncode, 0)
                report = json.loads((directory/'report.json').read_text())
                self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
