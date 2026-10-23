#!/usr/bin/env python3
"""
ARIA Web Server — Lightweight, zero-dependency REST API and dashboard server.

Serves the interactive web dashboard and exposes endpoints to solve problems,
retrieve real-time meta-learning statistics, trigger self-training, and inspect knowledge.

Usage:
    python server.py [--port 8000] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Windows console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add parent directory to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ai_reasoning_engine.core.engine import ARIAEngine
from ai_reasoning_engine.data.real_world_datasets import (
    PHYSICAL_CONSTANTS,
    FINANCIAL_DATA,
    REAL_WORLD_FLIGHT_DISTANCES,
    REAL_WORLD_BENCHMARKS,
)

# Global engine instance
_DATA_DIR = os.path.join(_PROJECT_ROOT, "aria_data")
ENGINE = ARIAEngine(data_dir=_DATA_DIR)
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class ARIAHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Handles REST API and static asset requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self) -> None:
        """Add permissive CORS and caching headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to API or static dashboard."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/api/stats":
            self._handle_get_stats()
        elif path == "/api/history":
            self._handle_get_history()
        elif path == "/api/knowledge":
            self._handle_get_knowledge()
        elif path == "/api/datasets":
            self._handle_get_datasets()
        elif path == "/api/health":
            self._send_json({"status": "healthy", "service": "ARIA Reasoning Engine"})
        elif path == "/" or path == "":
            self.path = "/index.html"
            super().do_GET()
        else:
            # Fallback to static file serving
            super().do_GET()

    def do_POST(self) -> None:
        """Route POST requests to reasoning actions."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_error(HTTPStatus.BAD_REQUEST, f"Malformed JSON: {e}")
            return

        if path == "/api/solve":
            self._handle_post_solve(data)
        elif path == "/api/train":
            self._handle_post_train()
        elif path == "/api/reset":
            self._handle_post_reset()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, f"Endpoint {path} not found")

    # ------------------------------------------------------------------ #
    #                          API Handlers                               #
    # ------------------------------------------------------------------ #

    def _handle_post_solve(self, data: dict) -> None:
        problem_text = data.get("problem", "").strip()
        if not problem_text:
            self._send_error(HTTPStatus.BAD_REQUEST, "Field 'problem' is required.")
            return

        try:
            result = ENGINE.solve_detailed(problem_text)
            self._send_json(result)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Engine failure: {exc}")

    def _handle_get_stats(self) -> None:
        try:
            stats = ENGINE.get_stats()
            self._send_json(stats)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Stats error: {exc}")

    def _handle_post_train(self) -> None:
        try:
            report = ENGINE.train()
            stats = ENGINE.get_stats()
            self._send_json({"report": report, "stats": stats})
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Training error: {exc}")

    def _handle_get_history(self) -> None:
        try:
            recent = ENGINE._tracker.get_recent(50)
            self._send_json({"history": recent})
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"History error: {exc}")

    def _handle_get_knowledge(self) -> None:
        try:
            items = ENGINE._knowledge.get_all()
            self._send_json({"knowledge": items, "count": len(items)})
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Knowledge error: {exc}")

    def _handle_get_datasets(self) -> None:
        try:
            flights = [
                {"origin": k[0], "destination": k[1], "distance_km": v}
                for k, v in REAL_WORLD_FLIGHT_DISTANCES.items()
            ]
            self._send_json({
                "physical_constants": PHYSICAL_CONSTANTS,
                "financial_data": FINANCIAL_DATA,
                "flight_routes": flights,
                "benchmarks_count": len(REAL_WORLD_BENCHMARKS),
            })
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Dataset error: {exc}")

    def _handle_post_reset(self) -> None:
        try:
            ENGINE._selector.reset()
            self._send_json({"status": "reset_successful", "message": "Algorithm selector weights reset."})
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Reset error: {exc}")

    # ------------------------------------------------------------------ #
    #                         Response Helpers                            #
    # ------------------------------------------------------------------ #

    def _send_json(self, payload: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": True, "message": message, "status": status.value}, status=status)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ARIAHTTPRequestHandler)
    print(f"\n=======================================================")
    print(f"  🧠 ARIA Interactive Dashboard Server Running")
    print(f"  URL: http://localhost:{port}")
    print(f"  Network: http://{host}:{port}")
    print(f"  API Docs:")
    print(f"    • POST /api/solve   -> Submit problem & get 1 perfect solution")
    print(f"    • GET  /api/stats   -> Live meta-learning analytics & trend")
    print(f"    • POST /api/train   -> Autonomous self-training cycle")
    print(f"    • GET  /api/history -> Recent reasoning trajectories")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ARIA server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
