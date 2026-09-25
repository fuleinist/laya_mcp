#!/usr/bin/env python
"""browser_act_probe — measurements for `laya_browser_act` on real pages and a synthetic board.

This is the evidence probe for the browser backend (PR #5). It drives the *worker process* the
same way the server does — spawn `laya_browser_worker.py` in the SDK venv, wait for the readiness
line (which covers the checkpoint load), then one JSON request per line — so what it measures is
the shipped path minus the MCP transport (`tests/smoke_mcp.py` covers that hop).

Four modes:

    --sample                 the release's own sample request (offline, 3 candidates).
                             Ground truth is known: the goal asks for a text query, the release's
                             sample expects TYPE_TEXT on the searchbox.
    --website URL            fetch a real page, extract its interactive elements, ask one question.
                             `--expect-label` asserts the target element by label, when the page has
                             an unambiguous right answer.
    --input FILE             one decision from a JSON request file (goal/page/elements), which is how
                             a captured accessibility tree or DOM snapshot is replayed.
    --game [N]               a synthetic DOM board: N rounds, each with a hint colour and nine colour
                             tiles, one of which matches. The correct index is known per round, so this
                             reports accuracy rather than an opinion. It is a *board of buttons*, not a
                             real game: nothing here is canvas or pixel-based.

    python probes/browser_act_probe.py --sample
    python probes/browser_act_probe.py --website https://example.com --expect-label "More information"
    python probes/browser_act_probe.py --game 6

Requires LAYA_BROWSER_DIR (and, if the SDK venv is not `<dir>/../.venv`, LAYA_BROWSER_PYTHON). Run it
with the *server's* interpreter — it only talks to the worker over stdio and never imports torch.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKER = ROOT / "laya_browser_worker.py"

SAMPLE = {
    "goal": "Search Wikipedia for 'Python programming language' and open the article about "
            "the Python language.",
    "page": {"url": "https://en.wikipedia.org/wiki/Main_Page",
             "title": "Wikipedia, the free encyclopedia",
             "text": "Wikipedia — The Free Encyclopedia. From today's featured article: ..."},
    "elements": [{"label": "Wikipedia The Free Encyclopedia", "role": "link"},
                 {"label": "Open Search Wikipedia", "role": "searchbox"},
                 {"label": "Search", "role": "button"}],
}

ROLE_BY_TAG = {
    "a": "link", "button": "button", "select": "combobox", "textarea": "textbox",
    "option": "option", "label": "label",
}
ROLE_BY_INPUT = {"text": "textbox", "search": "searchbox", "email": "textbox", "url": "textbox",
                 "tel": "textbox", "number": "spinbutton", "password": "textbox", "submit": "button",
                 "button": "button", "checkbox": "checkbox", "radio": "radio"}
INTERACTIVE = set(ROLE_BY_TAG) | {"input"}


class _Extract(HTMLParser):
    """Pull the interactive elements and the visible text out of a static page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[dict] = []
        self.text: list[str] = []
        self._open: list[tuple[str, int]] = []   # (tag, index into elements) for the innermost open
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "noscript"):
            self._skip += 1
            return
        if tag not in INTERACTIVE:
            return
        role = ROLE_BY_TAG.get(tag) or ROLE_BY_INPUT.get((attrs.get("type") or "text").lower(), "input")
        label = (attrs.get("aria-label") or attrs.get("placeholder") or attrs.get("value")
                 or attrs.get("title") or attrs.get("name") or "")
        self.elements.append({"label": " ".join(label.split()),
                              "role": role, "name": attrs.get("name") or ""})
        self._open.append((tag, len(self.elements) - 1))

    def handle_data(self, data):
        if self._skip:
            return
        text = " ".join(data.split())
        if not text:
            return
        self.text.append(text)
        if self._open:  # text inside a link/button is that element's label
            tag, index = self._open[-1]
            element = self.elements[index]
            element["label"] = (element["label"] + " " + text).strip() if element["label"] else text

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
            return
        if self._open and self._open[-1][0] == tag:
            self._open.pop()


def extract(html: str) -> tuple[list[dict], str]:
    parser = _Extract()
    parser.feed(html)
    text = " ".join(parser.text)
    text = re.sub(r"\s+", " ", text).strip()[:4000]
    return parser.elements, text


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "laya-mcp-browser-probe/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


class Worker:
    """The same stdio protocol the MCP server's LayaBrowser client speaks."""

    def __init__(self) -> None:
        ckpt = (os.environ.get("LAYA_BROWSER_DIR") or "").strip()
        if not ckpt:
            raise SystemExit("set LAYA_BROWSER_DIR to the browser checkpoint directory")
        python = (os.environ.get("LAYA_BROWSER_PYTHON") or "").strip()
        if not python:
            base = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
            python = os.path.join(base, ".venv", "Scripts", "python.exe")
            if not os.path.exists(python):
                python = os.path.join(base, ".venv", "bin", "python")
        if not os.path.exists(python):
            raise SystemExit(f"browser SDK python not found: {python!r}; set LAYA_BROWSER_PYTHON")
        self.load_s: float | None = None
        self.temperature = None
        self.proc = subprocess.Popen(
            [python, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        started = time.time()
        line = self.proc.stdout.readline()
        try:
            ready = json.loads(line)
        except Exception:
            raise SystemExit(f"worker did not report readiness: {line[:200]!r}")
        if ready.get("status") != "ready":
            raise SystemExit(f"worker failed to load the checkpoint: {ready.get('error')}")
        self.load_s = ready.get("load_s")
        self.temperature = ready.get("temperature")
        self.wall_load_s = time.time() - started

    def ask(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        try:
            self.proc.stdin.write('{"command": "stop"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def show(label: str, out: dict, expect_label: str | None = None) -> bool:
    operation = (out.get("operation") or {}).get("choice")
    probabilities = (out.get("operation") or {}).get("probabilities") or {}
    top = "/".join(f"{k} {v:.3f}" for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1])[:3])
    target = out.get("target_id")
    target_label = None
    if target:
        elements = out.get("_elements") or []
        if 0 < target <= len(elements):
            target_label = elements[target - 1].get("label")
    print(f"  {label}")
    print(f"    operation  {operation} (conf {(out.get('operation') or {}).get('confidence')})  "
          f"[{top}]")
    print(f"    target     [{target}] {'by ' + str(out.get('target_of')) if target else '(none)'}"
          + (f"  {target_label!r}" if target_label else ""))
    print(f"    latency    {out.get('ms')} ms   elements {out.get('elements_offered')}   "
          f"page chars {out.get('page_chars_in')}   output tokens "
          f"{(out.get('usage') or {}).get('output_tokens')}")
    if expect_label:
        hit = bool(target_label) and expect_label.lower() in (target_label or "").lower()
        print(f"    expectation {'MET' if hit else 'MISSED'}: target label should contain "
              f"{expect_label!r}, got {target_label!r}")
        return hit
    return True


def play_game(worker: Worker, rounds: int) -> None:
    """A synthetic DOM board: nine colour tiles, a hint colour, one correct index per round."""
    colours = ["red", "green", "blue", "yellow", "purple", "orange", "teal", "pink", "brown"]
    hits, misses, latencies = 0, [], []
    print(f"  synthetic DOM board: {rounds} rounds, 9 colour tiles, hint in the page text")
    for round_index in range(rounds):
        target_colour = colours[round_index % len(colours)]
        tiles = [c for c in colours]
        board = "".join(f'<button class="tile">{c.capitalize()} tile</button>' for c in tiles)
        html = (f"<html><head><title>Colour board {round_index + 1}</title></head><body>"
                f"<h1>Colour board</h1><p>Marker colour: {target_colour}</p>"
                f"<div id='board'>{board}</div></body></html>")
        elements, text = extract(html)
        request = {"goal": f"The board is showing a {target_colour} marker. Click the tile whose "
                           f"colour matches the marker, then stop.",
                   "page": {"url": f"file:///board/{round_index + 1}", "title": f"Colour board {round_index + 1}",
                            "text": text},
                   "elements": elements}
        out = worker.ask(request)
        out["_elements"] = elements
        chosen = (out.get("target") or {}).get("choice")
        picked = colours[int(chosen) - 1] if chosen and str(chosen).isdigit() and 1 <= int(chosen) <= 9 else None
        ok = picked == target_colour
        hits += 1 if ok else 0
        if not ok:
            misses.append((round_index + 1, target_colour, picked, out.get("ms")))
        if out.get("ms"):
            latencies.append(out["ms"])
        print(f"    round {round_index + 1}: hint {target_colour:7s} -> picked {str(picked):7s} "
              f"[{chosen}] {'ok' if ok else 'MISS'} in {out.get('ms')} ms")
    print(f"  accuracy {hits}/{rounds} ({hits / rounds:.3f}); "
          f"misses {misses or 'none'}; latency p50 "
          f"{statistics.median(latencies) if latencies else float('nan'):.0f} ms")
    print("  NOTE: a board of buttons, not a real game — canvas or pixel state is invisible to this")
    print("        backend by design (it answers with an index into the element list you pass).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="the release's own sample request")
    parser.add_argument("--website", metavar="URL", help="a real page: fetch, extract, ask")
    parser.add_argument("--input", metavar="FILE", help="a JSON request file to replay")
    parser.add_argument("--game", nargs="?", const=6, type=int, metavar="N",
                        help="synthetic DOM board, N rounds (default 6)")
    parser.add_argument("--expect-label", help="assert the chosen element's label contains this text")
    parser.add_argument("--goal", help="the goal to ask with (defaults per mode)")
    parser.add_argument("--rules", help="override the checkpoint's built-in rules")
    args = parser.parse_args()

    worker = Worker()
    print(f"worker ready: load {worker.load_s} s (handshake {worker.wall_load_s:.1f} s), "
          f"temperature {worker.temperature}")
    ok = True
    try:
        if args.website:
            html = fetch(args.website)
            elements, text = extract(html)
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
            request = {"goal": args.goal or (f"Open the link that explains what "
                                             f"{args.website.split('//')[-1].split('/')[0]} is, so I "
                                             f"can read about the site."),
                       "page": {"url": args.website,
                                "title": " ".join((title.group(1) if title else "").split()),
                                "text": text},
                       "elements": elements[:96]}
            if args.rules:
                request["rules"] = args.rules
            out = worker.ask(request)
            out["_elements"] = elements[:96]
            print(f"\n== website: {args.website} ({len(elements)} interactive elements found) ==")
            ok = show("goal: open the link that explains the site", out, args.expect_label) and ok
        elif args.input:
            request = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
            out = worker.ask(request)
            out["_elements"] = request.get("elements") or []
            print(f"\n== replay: {args.input} ==")
            ok = show(request.get("goal", ""), out, args.expect_label) and ok
        elif args.game:
            print("\n== game ==")
            play_game(worker, args.game)
        else:
            out = worker.ask(SAMPLE)
            out["_elements"] = SAMPLE["elements"]
            print("\n== the release's own sample request ==")
            ok = show("goal: search Wikipedia and open the Python article", out, "search") and ok
    finally:
        worker.close()
    if args.expect_label and not ok:
        print("\nFAILED: the target did not match the expected element")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())