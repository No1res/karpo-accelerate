import io
import json
from contextlib import contextmanager, redirect_stdout
from functools import partial
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest import TestCase

from karpo_accel.benchmark import arrivals, percentile, workload
from karpo_accel.cases import read_cases
from karpo_accel.cli import main
from karpo_accel.proxy import Handler as Gateway
from karpo_accel.routing import Replicas

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@contextmanager
def listener(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as service:
        worker = Thread(target=service.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield service.server_port
        finally:
            service.shutdown()
            worker.join()


class Replica(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def send(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send({"ok": True})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = request["state"]
        choice = "billing" if "invoice" in text or "payments" in text else "support"
        self.send({"answers": {key: {"choice": choice} for key in request["questions"]},
                   "replica": self.server.server_port})


class HTTPTests(TestCase):
    def test_inflight_selection_and_release(self):
        router = Replicas(["http://127.0.0.1:1", "http://127.0.0.1:2"], 1)
        with router.reserve() as first:
            self.assertEqual(first, 0)
            with self.assertRaises(RuntimeError), router.reserve() as second:
                self.assertEqual(second, 1)
                raise RuntimeError("request failed")
            self.assertEqual(router.inflight, [1, 0])
        self.assertEqual(router.inflight, [0, 0])

    def test_gateway_routes_choice_requests(self):
        router = None
        with listener(Replica) as first, listener(Replica) as second:
            router = Replicas([f"http://127.0.0.1:{p}" for p in (first, second)], 2)
            with listener(partial(Gateway, replicas=router, max_body_bytes=1024)) as port:
                connection = HTTPConnection("127.0.0.1", port, timeout=3)
                try:
                    connection.request("GET", "/healthz")
                    response = connection.getresponse()
                    self.assertEqual(json.loads(response.read()), {"ok": True, "replicas": [True, True]})
                    for p in (first, second):
                        connection.request("POST", "/v1/systemone", (EXAMPLES / "request.json").read_bytes())
                        response = connection.getresponse()
                        body = json.loads(response.read())
                        self.assertEqual(response.status, 200)
                        self.assertEqual(body["replica"], p)
                        self.assertEqual(body["answers"]["decision"]["choice"], "billing")
                finally:
                    connection.close()
        self.assertEqual(router.inflight, [0, 0])

    def test_open_loop_cli_reports_latency_and_accuracy(self):
        with listener(Replica) as first, listener(Replica) as second:
            output = io.StringIO()
            with redirect_stdout(output):
                main(["bench", "--replica", f"http://127.0.0.1:{first}", "--replica", f"http://127.0.0.1:{second}",
                      "--requests", str(EXAMPLES / "requests.jsonl"), "--qps", "20", "--seconds", "0.2",
                      "--arrivals", "poisson"])
            result = json.loads(output.getvalue())
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["accuracy"], 1)
        self.assertGreater(result["http_p95_ms"], 0)
        self.assertGreaterEqual(result["arrival_to_response_p95_ms"], result["http_p95_ms"])

    def test_open_loop_uses_distinct_contexts_and_reproducible_arrivals(self):
        cases = read_cases(EXAMPLES / "requests.jsonl", labels=True)
        self.assertEqual(len(workload(cases + cases, 4, 42, repeat=False)), 4)
        with self.assertRaisesRegex(ValueError, "distinct contexts"):
            workload(cases, 5, 42, repeat=False)
        self.assertEqual(arrivals("poisson", 4, 2, 42), arrivals("poisson", 4, 2, 42))
        self.assertEqual(percentile([3, 1, 2], 0.95), 2.9)
