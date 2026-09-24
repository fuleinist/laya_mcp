"""Step 3 of the computer-use integration: verify a step from a real accessibility diff.

`docs/computer-use.md` §4.2 argues the loop should send the tree once and then only the deltas, and
this is the measurement of that argument on the captures already sitting on this machine, rather than
on a synthetic diff. The questions and their gold answers come from `laya_router.a11y`; this module
runs backends over them and scores the results the same way `eval.py` scores routing:

* per-kind accuracy with a Wilson interval, **and the majority-class baseline for that kind**, so a
  question a constant answer would ace cannot look like a capability;
* latency and the engine's own input-token count against the ~768-token state budget, because "a diff
  fits the window" is a claim about a distribution, not one number;
* exact McNemar against a hosted backend on the same items, so "good enough" is a comparison.

Only metadata is written to the results file — kinds, gold and predicted values, roles, sizes. The
screen text itself (labels) stays on the machine; `--keep-raw` additionally stores the raw daemon
replies, which do carry labels, and is off by default.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from laya_router import a11y, backends
from laya_router.env import load_env_file
from laya_router.metrics import paired, percentile, summarise_single, wilson

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"
STATE_BUDGET_TOKENS = 768


# --- running -----------------------------------------------------------------

def build_items(pairs: list[dict[str, Any]], max_chars: int = 1600, variant: str = "labels",
                max_questions: int | None = None) -> list[dict[str, Any]]:
    """One work item per capture pair: its diff, the questions, and the gold answers."""
    work = []
    for i, pair in enumerate(pairs):
        diff_result = a11y.diff(pair["before"], pair["after"], variant=variant)
        if len(diff_result["text"]) > max_chars:
            diff_result = _cap(diff_result, max_chars, variant)
        items = a11y.build_questions(pair, diff_result, pair_index=i)
        if max_questions:
            items = items[:max_questions]
        work.append({"pair": i, "app": pair["app"], "diff": diff_result, "items": items,
                     "before_path": pair["before_path"], "after_path": pair["after_path"]})
    return work


def _cap(diff_result: dict[str, Any], max_chars: int, variant: str = "labels") -> dict[str, Any]:
    """Trim a rendered diff to a character budget, keeping whole lines and the + / - split.

    A window repaint (a whole app opening) produces a diff far past the state budget. Trimming by
    lines rather than characters keeps every line a complete `<role>: <label>` pair, and the flag
    travels with the result so a truncated reading is never mistaken for a complete one.
    """
    half = max(1, max_chars // 2)
    added, removed, used_a, used_r = [], [], 0, 0
    for line in diff_result["kept_added"]:
        if used_a + len(line) + 2 > half:
            break
        added.append(line)
        used_a += len(line) + 2
    for line in diff_result["kept_removed"]:
        if used_r + len(line) + 2 > half:
            break
        removed.append(line)
        used_r += len(line) + 2
    out = dict(diff_result)
    out.update({"kept_added": added, "kept_removed": removed, "truncated": True,
                "text": a11y.render_diff_text(added, removed, variant), "variant": variant})
    out["lines"] = len(added) + len(removed)
    out["chars"] = len(out["text"])
    out["approx_tokens"] = round(out["chars"] / 2.2)
    out["capped_at_chars"] = max_chars
    return out


def run_backend(backend: backends.Backend, work: list[dict[str, Any]], timeout_s: float,
                progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Ask one backend every question of every pair; one row per question, in order."""
    rows: list[dict[str, Any]] = []
    for entry in work:
        result = a11y.verify(backend, entry["diff"], entry["items"], timeout_s)
        answers = result["answers"]
        for item in entry["items"]:
            pred = answers.get(item["name"])
            rows.append({
                "pair": entry["pair"], "app": entry["app"], "kind": item["kind"],
                "name": item["name"], "type": item["type"], "role": item.get("role"),
                "gold": item["gold"], "pred": pred, "correct": pred == item["gold"],
                "answered": result["answered"], "latency_ms": result["latency_ms"],
                "input_tokens": result["input_tokens"], "diff_chars": entry["diff"]["chars"],
                "diff_lines": entry["diff"]["lines"], "diff_truncated": entry["diff"]["truncated"],
                "variant": entry["diff"].get("variant", "prefix"),
                "error": result["error"],
            })
        if progress:
            progress(f"  pair {entry['pair']:02d} {entry['app'][:20]:20s} "
                     f"answered={result['answered']} diff {entry['diff']['chars']}c "
                     f"~{entry['diff']['approx_tokens']}tok")
    return rows


# --- scoring -----------------------------------------------------------------

def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-kind accuracy with its baseline, plus an overall count and a size profile."""
    out: dict[str, Any] = {"overall": _score_rows(rows), "by_kind": {}}
    kinds = sorted({r["kind"] for r in rows})
    for kind in kinds:
        subset = [r for r in rows if r["kind"] == kind]
        scored = _score_rows(subset)
        counts: dict[str, int] = {}
        for row in subset:
            counts[str(row["gold"])] = counts.get(str(row["gold"]), 0) + 1
        # The baseline a constant answer gets: this is what separates a capability from a question
        # whose gold is nearly always the same value.
        scored["majority_baseline"] = (max(counts.values()) / len(subset)) if subset else None
        scored["gold_counts"] = counts
        out["by_kind"][kind] = scored
    sizes = [r["diff_chars"] for r in rows]
    tokens = [r["input_tokens"] for r in rows if r["input_tokens"]]
    out["size"] = {
        "diff_chars": {"p50": percentile(sizes, 0.5), "p90": percentile(sizes, 0.9),
                       "max": max(sizes) if sizes else None},
        "engine_input_tokens": {"p50": percentile(tokens, 0.5), "p90": percentile(tokens, 0.9),
                                "max": max(tokens) if tokens else None},
        "over_budget_share": (sum(1 for t in tokens if t > STATE_BUDGET_TOKENS) / len(tokens))
        if tokens else None,
        "truncated_share": sum(1 for r in rows if r["diff_truncated"]) / len(rows) if rows else None,
        "state_budget_tokens": STATE_BUDGET_TOKENS,
    }
    out["latency_ms"] = {"p50": percentile([r["latency_ms"] for r in rows], 0.5),
                         "p99": percentile([r["latency_ms"] for r in rows], 0.99)}
    out["failures"] = sum(1 for r in rows if not r["answered"])
    return out


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": None, "accuracy_ci95": [0.0, 1.0]}
    gold = [r["gold"] for r in rows]
    pred = [None if r["pred"] is None else r["pred"] for r in rows]
    labels = sorted({str(v) for v in gold})
    scored = summarise_single(gold, pred, labels,
                              latencies=[r["latency_ms"] for r in rows])
    scored["accuracy_ci95"] = list(wilson(scored["correct"], scored["n"]))
    return scored


def compare(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]],
            a_name: str, b_name: str) -> dict[str, Any]:
    """Paired comparison on the items both backends answered, and on all items."""
    by_name_a = {r["name"] + str(r["pair"]): r for r in rows_a}
    shared = [r for r in rows_b if r["name"] + str(r["pair"]) in by_name_a]
    gold = [r["gold"] for r in shared]
    labels = sorted({str(v) for v in gold})
    both = paired(gold, [by_name_a[r["name"] + str(r["pair"])]["pred"] for r in shared],
                  [r["pred"] for r in shared], labels)
    both["a"] = a_name
    both["b"] = b_name
    both["scope"] = "items asked of both backends; an unanswered item counts as a miss"
    return both


# --- report ------------------------------------------------------------------

def render_markdown(result: dict[str, Any]) -> str:
    lines = [f"# verify_step on real accessibility diffs ({result['schema']})", "",
             f"captures: {result['captures']} pairs from {result['capture_dir']}", ""]
    for name, block in result["metrics"].items():
        lines += [f"## {name}", "", f"overall: {_pct(block['overall']['accuracy'])} "
                  f"({block['overall']['correct']}/{block['overall']['n']}, "
                  f"95% CI {_pct(block['overall']['accuracy_ci95'][0])}-"
                  f"{_pct(block['overall']['accuracy_ci95'][1])}), failures {block['failures']}", "",
                  "| kind | n | accuracy | majority baseline | gold split |", "|---|---:|---:|---:|---|"]
        for kind, k in block["by_kind"].items():
            lines.append(f"| {kind} | {k['n']} | {_pct(k['accuracy'])} | "
                         f"{_pct(k['majority_baseline'])} | {k['gold_counts']} |")
        size = block["size"]
        lines += ["", f"diff chars p50 {size['diff_chars']['p50']:.0f} / p90 "
                  f"{size['diff_chars']['p90']:.0f} / max {size['diff_chars']['max']:.0f}; "
                  f"engine input tokens p50 {_num(size['engine_input_tokens']['p50'])} / p90 "
                  f"{_num(size['engine_input_tokens']['p90'])} / max "
                  f"{_num(size['engine_input_tokens']['max'])} against a "
                  f"{size['state_budget_tokens']}-token budget "
                  f"(over budget {_pct(size['over_budget_share'])}, truncated "
                  f"{_pct(size['truncated_share'])}); latency p50 "
                  f"{block['latency_ms']['p50']:.0f} ms, p99 {block['latency_ms']['p99']:.0f} ms", ""]
    if result.get("paired"):
        p = result["paired"]
        lines += [f"## paired: {p['a']} vs {p['b']} ({p['scope']})", "",
                  f"{p['both_correct']} both, {p['a_only_correct']} {p['a']} only, "
                  f"{p['b_only_correct']} {p['b']} only, {p['neither_correct']} neither; "
                  f"McNemar exact p = {p['mcnemar_exact_p']:.3f}", ""]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


# --- cli ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Laya on real accessibility diffs")
    parser.add_argument("--captures", default=None, help="directory of cua-driver captures")
    parser.add_argument("--backends", default="laya",
                        help="comma-separated specs: laya, openai:<base_url>|<model>|<KEY_ENV>")
    parser.add_argument("--env-file", default=None, help="KEY=VALUE file to load, never echoed")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N pairs")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-chars", type=int, default=1600, help="diff character budget")
    parser.add_argument("--variant", default="labels", choices=list(a11y.DIFF_VARIANTS),
                        help="how the diff is serialised for the backend")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="cap the questions per pair (token attribution)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--keep-raw", action="store_true",
                        help="store raw daemon replies (they carry screen labels) in the results file")
    parser.add_argument("--markdown", default=None, help="write the report here as well")
    args = parser.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)
    captures = a11y.load_capture_dir(args.captures)
    pairs = a11y.pair_captures(captures)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        print(f"no capture pairs found in {args.captures or a11y.DEFAULT_CAPTURES}")
        return 2

    work = build_items(pairs, max_chars=args.max_chars, variant=args.variant,
                       max_questions=args.max_questions)
    specs = [s.strip() for s in args.backends.split(",") if s.strip()]
    registry = backends.default_registry()
    rows_by_backend: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {}
    for spec in specs:
        if spec in registry:
            backend = registry.pop(spec)  # pop, so an explicit spec cannot collide with an env one
        else:
            name = "frontier"
            while name in rows_by_backend:
                name += "2"
            backend = backends.parse_spec(spec, name=name)
        print(f"running {backend.name} over {len(pairs)} pairs "
              f"({sum(len(w['items']) for w in work)} questions)")
        t0 = time.perf_counter()
        rows = run_backend(backend, work, args.timeout, progress=print)
        rows_by_backend[backend.name] = rows
        metrics[backend.name] = score(rows)
        print(f"  {backend.name}: {time.perf_counter() - t0:.1f}s")

    result: dict[str, Any] = {
        "schema": "verify-v1", "variant": args.variant, "max_chars": args.max_chars,
        "capture_dir": str(args.captures or a11y.DEFAULT_CAPTURES),
        "captures": len(captures), "pairs": len(pairs),
        "pairs_detail": [{"pair": w["pair"], "app": w["app"], "chars": w["diff"]["chars"],
                          "lines": w["diff"]["lines"], "truncated": w["diff"]["truncated"],
                          "questions": len(w["items"])} for w in work],
        "metrics": metrics,
        "rows": [{k: v for k, v in r.items() if k != "raw"} for rows in rows_by_backend.values()
                 for r in rows],
    }
    if len(rows_by_backend) == 2:
        (a_name, a_rows), (b_name, b_rows) = rows_by_backend.items()
        result["paired"] = compare(a_rows, b_rows, a_name, b_name)
    if args.keep_raw:
        result["raw"] = {name: [r.get("raw") for r in rows] for name, rows in rows_by_backend.items()}

    out = Path(args.out) if args.out else RESULTS_DIR / f"verify_{'_vs_'.join(metrics)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_markdown(result)
    print("\n" + report)
    if args.markdown:
        Path(args.markdown).write_text(report + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())