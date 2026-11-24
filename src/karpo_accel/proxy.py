"""A local HTTP gateway for independently launched Karpo replicas."""

import json
from functools import partial
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .routing import Replicas


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60
    disable_nagle_algorithm = True

    def __init__(self, *args, replicas, max_body_bytes, **kwargs):
        self.replicas = replicas
        self.max_body_bytes = max_body_bytes
        super().__init__(*args, **kwargs)

    def finish(self):
        try:
            super().finish()
        finally:
            self.replicas.close_connections()

    def send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def reject(self, status, message):
        self.close_connection = True
        self.send(status, json.dumps({"error": {"message": message}}).encode())

    def do_POST(self):
        if urlsplit(self.path).path != "/v1/systemone":
            return self.reject(404, "not found")
        length = self.headers.get("Content-Length", "")
        if not length.isascii() or not length.isdigit() or len(length) > 12:
            return self.reject(411, "a valid Content-Length is required")
        if int(length) > self.max_body_bytes:
            return self.reject(413, "request body exceeds the gateway limit")
        body = self.rfile.read(int(length))
        if len(body) != int(length):
            return self.reject(400, "incomplete request body")
        headers = {name: self.headers[name] for name in ("Content-Type", "Content-Encoding") if name in self.headers}
        try:
            with self.replicas.reserve() as replica:
                response = self.replicas.request(replica, "POST", "/v1/systemone", body, headers)
        except (HTTPException, OSError):
            return self.reject(502, "Karpo replica did not return a response")
        self.send(*response)

    def do_GET(self):
        if urlsplit(self.path).path != "/healthz":
            return self.reject(404, "not found")
        live = []
        for slot in range(len(self.replicas.urls)):
            try:
                status, _, _ = self.replicas.request(slot, "GET", "/healthz")
                live.append(status == 200)
            except (HTTPException, OSError):
                live.append(False)
        self.send(200 if all(live) else 503, json.dumps({"ok": all(live), "replicas": live}).encode())


def run_proxy(args):
    replicas = Replicas(args.replica, args.timeout)
    handler = partial(Handler, replicas=replicas, max_body_bytes=args.max_body_bytes)
    with ThreadingHTTPServer((args.host, args.port), handler) as gateway:
        print(f"Karpo gateway listening on {args.host}:{args.port}", flush=True)
        gateway.serve_forever()
