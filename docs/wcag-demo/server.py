#!/usr/bin/env python3
"""Same-origin host for the WCAG demo: static files plus a /api proxy to vegaml-serve.

vegaml-serve itself has no CORS, so the page can't call it cross-origin from the browser.
This little server serves the demo and forwards /api/* to the API on the backend:

    vegaml-serve --model 0.8b --port 8080 &
    python3 server.py                # -> http://127.0.0.1:8090  (API defaults to http://127.0.0.1:8080)

Point the API elsewhere with API_TARGET=http://host:port python3 server.py.
Only stdlib — no dependencies.
"""
import http.server
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent
API_TARGET = os.environ.get("API_TARGET", "http://127.0.0.1:8080").rstrip("/")
PORT = int(os.environ.get("PORT", "8090"))

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".json": "application/json; charset=utf-8", ".css": "text/css; charset=utf-8"}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/"):  # GET health/model
            self.proxy(method="GET")
            return
        rel = path.lstrip("/") or "index.html"
        f = HERE / rel
        if not f.is_file():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, f.read_bytes(), CONTENT_TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.proxy(method="POST", body=body, ctype=self.headers.get("Content-Type", ""))

    def proxy(self, method, body=b"", ctype=""):
        # vegaml-serve's routes: GET /health, GET /v1/model, POST /v1/decide[/batch].
        # /api/health -> /health, everything else -> /v1 + the /api tail.
        tail = self.path[len("/api"):]
        url = API_TARGET + ("/health" if tail == "/health" else "/v1" + tail)
        req = urllib.request.Request(url, data=body if method == "POST" else None, method=method,
                                     headers={"Content-Type": ctype or "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = r.read()
                self._send(r.status, out, r.headers.get("Content-Type", "application/json"))
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode()
            self._send(502, msg, "application/json")

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    print(f"demo  -> http://127.0.0.1:{PORT}   api proxy -> {API_TARGET}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
