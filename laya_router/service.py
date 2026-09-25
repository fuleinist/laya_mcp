"""The router service: `GET /health`, `GET /questions`, `POST|GET /route`.

Step 2 of the computer-use integration (issue #3) needs `/route` reachable over HTTP, and the
same decision reachable as an MCP tool from both harnesses. This is the HTTP half: a stdlib
`ThreadingHTTPServer` (no new dependency), one shared question schema, and whichever backends are
configured answer it.

    python -m laya_router.service --port 8760 \
        --frontier 'openai:https://host/v1|model|KEY_ENV'

    curl 'http://127.0.0.1:8760/questions'
    curl 'http://127.0.0.1:8760/route?task=Cut+a+release+and+publish+the+artifacts'
    curl -X POST http://127.0.0.1:8760/route -d '{"task":"...","backend":"frontier"}'

`/route` answers `{"tier": ..., "needs_tools": ..., "sensitive": ..., "advisory": true, ...}`. It is
advisory: nothing here blocks an action, because step 1 measures whether the router is accurate
enough to be worth enforcing (see `docs/router-service.md`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from laya_router import __version__
from laya_router import questions as Q
from laya_router.backends import Backend, BackendError, default_registry
from laya_router.env import load_env_file

SERVICE = "laya-router"


class RouterState:
    """Everything a request handler needs: the schema, the backends, and a call counter."""

    def __init__(self, registry: dict[str, Backend] | None = None, schema: dict[str, Any] | None = None,
                 timeout_s: float = 60.0) -> None:
        self.schema = schema or Q.load_schema()
        self.registry = registry if registry is not None else default_registry()
        self.timeout_s = timeout_s
        self.started_at = time.time()
        self.calls = 0

    def pick(self, name: str | None) -> Backend:
        name = (name or "").strip() or "laya"
        if name == "auto":
            # "auto" means the first configured backend that is reachable — used by cron callers
            # that would rather fall back to the chat backend than fail on a cold engine.
            for candidate in self.registry.values():
                if candidate.health().get("reachable"):
                    return candidate
            return next(iter(self.registry.values()))
        if name not in self.registry:
            raise KeyError(f"unknown backend {name!r}; configured: {', '.join(self.registry) or 'none'}")
        return self.registry[name]

    def route(self, task: str, context: str | None = None, backend: str | None = None) -> dict[str, Any]:
        state = Q.build_state(task, context)
        chosen = self.pick(backend)
        self.calls += 1
        decision = chosen.route(state, self.schema, self.timeout_s)
        body = decision.to_dict()
        body |= {
            "service": SERVICE,
            "schema_version": self.schema["schema_version"],
            "schema_digest": Q.digest(self.schema),
            "advisory": True,
            "boundary": "advisory only, and a text-channel decision cannot see image-embedded "
                        "instructions in the screen (see docs/computer-use.md section 6)",
        }
        return body


class RouterHandler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE}/{__version__}"
    state: RouterState  # injected by `serve`

    # ---- helpers ---------------------------------------------------------
    def _send(self, code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False, indent=1).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code: int, message: str, **extra: Any) -> None:
        self._send(code, {"error": message, "status": code, **extra})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the default line, on stderr
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/health", "/healthz"):
            return self._health()
        if parsed.path == "/questions":
            return self._questions()
        if parsed.path == "/route":
            task = (query.get("task") or [""])[0]
            context = (query.get("context") or [None])[0]
            backend = (query.get("backend") or [None])[0]
            return self._route(task, context, backend)
        if parsed.path == "/":
            return self._send(200, {"service": SERVICE, "version": __version__,
                                    "endpoints": ["GET /health", "GET /questions",
                                                  "POST /route", "GET /route?task=..."]})
        return self._error(404, f"no such path: {parsed.path}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._body()
        except ValueError as exc:
            return self._error(400, str(exc))
        if parsed.path == "/route":
            return self._route(body.get("task", ""), body.get("context"), body.get("backend"))
        if parsed.path == "/questions":
            return self._questions()
        return self._error(404, f"no such path: {parsed.path}")

    # ---- handlers --------------------------------------------------------
    def _health(self) -> None:
        state = self.state
        backends = {name: backend.health() for name, backend in state.registry.items()}
        reachable = any(b.get("reachable") for b in backends.values())
        self._send(200 if reachable else 503, {
            "ok": reachable,
            "service": SERVICE,
            "version": __version__,
            "schema_version": state.schema["schema_version"],
            "schema_digest": Q.digest(state.schema),
            "tier_options": Q.tier_options(state.schema),
            "backends": backends,
            "uptime_s": round(time.time() - state.started_at, 1),
            "calls": state.calls,
            "advisory": True,
        })

    def _questions(self) -> None:
        state = self.state
        self._send(200, {
            "schema_version": state.schema["schema_version"],
            "digest": Q.digest(state.schema),
            "notes": state.schema.get("notes"),
            "state_hint": state.schema.get("state_hint"),
            "questions": Q.questions_of(state.schema),
            "tier_options": Q.tier_options(state.schema),
            "render_example": Q.render_prompt(state.schema, {"task": "<the step description>"}),
            "backends": sorted(state.registry),
        })

    def _route(self, task: Any, context: Any, backend: Any) -> None:
        if not isinstance(task, str) or not task.strip():
            return self._error(400, "`task` is required and must be a non-empty string",
                               hint="POST /route {\"task\": \"...\"} or GET /route?task=...")
        if context is not None and not isinstance(context, str):
            return self._error(400, "`context` must be a string when present")
        if backend is not None and not isinstance(backend, str):
            return self._error(400, "`backend` must be a string when present")
        try:
            body = self.state.route(task, context, backend)
        except KeyError as exc:
            return self._error(400, str(exc).strip("'"))
        except BackendError as exc:
            return self._error(502, str(exc))
        except ValueError as exc:
            return self._error(400, str(exc))
        if not body.get("answered"):
            # The call itself failed (engine down, key missing, unparseable reply). Say so with a
            # gateway status rather than a 200 that reads like a decision.
            return self._send(502, body)
        return self._send(200, body)


def serve(registry: dict[str, Backend] | None = None, host: str = "127.0.0.1", port: int = 8760,
          schema: dict[str, Any] | None = None, timeout_s: float = 60.0) -> ThreadingHTTPServer:
    """Build (but do not start) the server. `port=0` picks a free port, which tests rely on."""
    state = RouterState(registry=registry, schema=schema, timeout_s=timeout_s)
    handler = type("BoundRouterHandler", (RouterHandler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.state = state  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Router service: /health, /questions, /route")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--schema", default="", help="question schema JSON (default: data/questions.json)")
    ap.add_argument("--frontier", default="", help="openai:<base_url>|<model>|<KEY_ENV>; empty = Laya only")
    ap.add_argument("--env-file", default="", help="file of KEY=VALUE lines to load first, so a key "
                                                   "never has to appear in a command line or log")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)

    schema = Q.load_schema(args.schema or None)
    registry = default_registry(args.frontier or None)
    httpd = serve(registry=registry, host=args.host, port=args.port, schema=schema,
                  timeout_s=args.timeout)
    host, port = httpd.server_address[:2]
    print(f"{SERVICE} {__version__} on http://{host}:{port}  schema {schema['schema_version']} "
          f"({Q.digest(schema)})  backends: {', '.join(registry)}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())