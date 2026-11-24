"""Send each request to the replica with the fewest outstanding requests."""

from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from threading import Lock, local
from time import monotonic
from urllib.parse import urlsplit


class Replicas:
    def __init__(self, urls, timeout):
        self.urls = tuple(urlsplit(url) for url in urls)
        if not self.urls or any(
            url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.path not in ("", "/") or url.query or url.fragment
            for url in self.urls
        ):
            raise ValueError("Replica URLs must be HTTP(S) origins such as http://127.0.0.1:18440")
        for url in self.urls:
            if url.port == 0:
                raise ValueError("Replica ports must be between 1 and 65535")
        self.timeout = timeout
        self.inflight = [0] * len(self.urls)
        self.turn = 0
        self.lock = Lock()
        self.local = local()
        self.connections = set()

    @contextmanager
    def reserve(self):
        count = len(self.urls)
        with self.lock:
            slot = min(range(count), key=lambda i: (self.inflight[i], (i - self.turn) % count))
            self.inflight[slot] += 1
            self.turn = (slot + 1) % count
        try:
            yield slot
        finally:
            with self.lock:
                self.inflight[slot] -= 1

    def request(self, slot, method, path, body=None, headers=None):
        if not hasattr(self.local, "connections"):
            self.local.connections = {}
            self.local.last_completed = {}
        connections = self.local.connections
        conn = connections.get(slot)
        if conn is None:
            url = self.urls[slot]
            cls = HTTPSConnection if url.scheme == "https" else HTTPConnection
            conn = connections[slot] = cls(url.hostname, url.port, timeout=self.timeout)
            with self.lock:
                self.connections.add(conn)
        elif monotonic() - self.local.last_completed.get(slot, 0.0) > 55:
            conn.close()
        try:
            conn.request(method, path, body, headers or {})
            response = conn.getresponse()
            body = response.read()
            self.local.last_completed[slot] = monotonic()
            return response.status, body, response.getheader("Content-Type", "application/json")
        except (HTTPException, OSError):
            conn.close()
            raise

    def close_connections(self):
        local_connections = getattr(self.local, "connections", {})
        with self.lock:
            self.connections.difference_update(local_connections.values())
        for connection in local_connections.values():
            connection.close()
        local_connections.clear()

    def close(self):
        with self.lock:
            connections, self.connections = self.connections, set()
        for connection in connections:
            connection.close()
