"""Step 2 evidence: `/route` reachable over HTTP and as an MCP tool, from the same schema.

Both harnesses on this machine (Hermes, OpenClaw) load MCP servers over stdio, so the MCP half of
this probe drives the server exactly as they do — as an MCP client — and the HTTP half starts the
`laya_router.service` in-process. The claim under test is not "it answers" but "both paths answer
the *same* question": the schema digest and the tier have to match, or a caller cannot compare an
HTTP decision with an MCP one.

    cd probes
    LAYA_EXE=/path/to/laya LAYA_MODEL=/path/to/model.gguf python route_step_probe.py

Set `LAYA_ROUTER_FRONTIER='openai:<base_url>|<model>|<KEY_ENV>'` and the HTTP half also exercises the
hosted backend, so the same question is shown answered by two different models through one schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.request

REPO_ROOT = os.environ.get("LAYA_MCP_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("LAYA_EXE", "G:/dev/AI/laya/layabin/laya.exe")
os.environ.setdefault("LAYA_MODEL", "G:/dev/AI/laya/laya_multilingual_q8_0.gguf")

from laya_router import questions as Q  # noqa: E402
from laya_router.service import serve  # noqa: E402


TASKS = [
    ("mechanical", "Bump five devDependencies in the workspace."),
    ("design", "Design the tier ladder and the escalation rule for the computer-use API."),
    ("sensitive", "Tag the verified commit and publish the release artifacts."),
]


def http_route(base: str, task: str, backend: str | None = None) -> dict:
    body = {"task": task}
    if backend:
        body["backend"] = backend
    req = urllib.request.Request(base + "/route", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def mcp_route(server: str, task: str, context: str | None = None) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[server], env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            assert "route_step" in names, f"route_step missing from {names}"
            args = {"task": task}
            if context:
                args["context"] = context
            result = await session.call_tool("route_step", args)
            text = "".join(c.text for c in result.content if getattr(c, "text", None))
            return json.loads(text)


async def main() -> int:
    server = os.path.join(REPO_ROOT, "laya_mcp_server.py")
    schema = Q.load_schema()
    digest = Q.digest(schema)

    registry_note = os.environ.get("LAYA_ROUTER_FRONTIER", "").strip() or "(not configured)"
    print(f"schema {schema['schema_version']} digest {digest}")
    print(f"frontier backend: {registry_note}\n")

    httpd = serve(host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    failures: list[str] = []
    try:
        health = json.loads(urllib.request.urlopen(base + "/health", timeout=30).read().decode())
        print(f"GET /health  ok={health['ok']} backends={sorted(health['backends'])} "
              f"schema={health['schema_digest']} calls={health['calls']}")
        questions = json.loads(urllib.request.urlopen(base + "/questions", timeout=30).read().decode())
        print(f"GET /questions  {len(questions['questions'])} questions, "
              f"options={questions['tier_options']} digest={questions['digest']}\n")

        print(f"{'task':10s} {'path':6s} {'backend':9s} {'tier':9s} {'p':>6s} {'tools':>6s} "
              f"{'sens':>6s} {'ms':>7s}  digest")
        for label, task in TASKS:
            t0 = time.time()
            over_http = http_route(base, task)
            http_ms = over_http.get("latency_ms") or (time.time() - t0) * 1000
            over_mcp = await mcp_route(server, task)
            for path, body, ms in (("HTTP", over_http, http_ms), ("MCP", over_mcp, None)):
                print(f"{label:10s} {path:6s} {body.get('backend', '?'):9s} {str(body.get('tier')):9s} "
                      f"{(body.get('tier_prob') or 0):6.3f} {str(body.get('needs_tools')):>6s} "
                      f"{str(body.get('sensitive')):>6s} "
                      f"{(ms if ms else body.get('latency_ms') or 0):7.0f}  {body.get('schema_digest')}")
            if over_http.get("schema_digest") != digest or over_mcp.get("schema_digest") != digest:
                failures.append(f"{label}: a path answered a different schema digest")
            if over_http.get("tier") != over_mcp.get("tier"):
                failures.append(
                    f"{label}: HTTP said {over_http.get('tier')}, MCP said {over_mcp.get('tier')} "
                    "for the same task and backend"
                )

        if registry_note != "(not configured)":
            print()
            for label, task in TASKS[:2]:
                hosted = http_route(base, task, backend="frontier")
                print(f"{label:10s} {'HTTP':6s} {'frontier':9s} {str(hosted.get('tier')):9s} "
                      f"{(hosted.get('tier_prob') or 0):6.3f} {str(hosted.get('needs_tools')):>6s} "
                      f"{str(hosted.get('sensitive')):>6s} {hosted.get('latency_ms') or 0:7.0f}  "
                      f"{hosted.get('schema_digest')}  configured={registry_note}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    print()
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK  both paths answered the same schema (digest and tier matched on every task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))