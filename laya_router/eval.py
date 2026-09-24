"""Measure the two backends on the committed corpus: accuracy with intervals, per-class recall,
exact McNemar over the discordant pairs, calibration, latency and tokens.

    python -m laya_router.eval --backends laya,openai:https://host/v1|model|KEY_ENV \
        --out laya_router/data/results/run.json

The corpus is `data/steps.jsonl`: real work items from this machine's own repositories and cron
schedules, plus the typed questions a computer-use loop asks. Every item carries gold labels for
the same three questions both backends answer, so the comparison is paired item-for-item.

Reported numbers are whatever the run produced. `--keep-raw` stores each backend's raw reply so a
disagreement can be read rather than guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from laya_router import metrics as M
from laya_router import questions as Q
from laya_router.backends import Backend, Decision, parse_spec
from laya_router.env import load_env_file

DATA = Path(__file__).resolve().parent / "data"
TIERS = ("economy", "frontier")


# --- corpus -----------------------------------------------------------------

def load_corpus(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Read and validate the labelled corpus."""
    path = Path(path) if path else DATA / "steps.jsonl"
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not JSON: {exc}") from exc
        for field in ("id", "task", "labels", "provenance"):
            if field not in item:
                raise ValueError(f"{path}:{lineno}: missing {field!r}")
        if item["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item['id']!r}")
        seen.add(item["id"])
        labels = item["labels"]
        if labels.get("tier") not in TIERS:
            raise ValueError(f"{path}:{lineno}: tier label must be one of {TIERS}")
        for flag in Q.FLAGS:
            if not isinstance(labels.get(flag), bool):
                raise ValueError(f"{path}:{lineno}: {flag} label must be a boolean")
        items.append(item)
    if not items:
        raise ValueError(f"{path}: empty corpus")
    return items


def corpus_digest(items: Iterable[dict[str, Any]]) -> str:
    blob = json.dumps(list(items), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --- running -----------------------------------------------------------------

def decision_from_dict(data: dict[str, Any]) -> Decision:
    """Rebuild a Decision from a stored prediction (used by `--resume`)."""
    fields = set(Decision.__dataclass_fields__)
    return Decision(**{k: v for k, v in data.items() if k in fields})


def run_backend(backend: Backend, items: list[dict[str, Any]], schema: dict[str, Any],
                timeout_s: float, retries: int = 2, sleep_s: float = 0.0,
                progress: bool = True, carried: list[Decision] | None = None) -> list[Decision]:
    """Route every item, retrying a *failed* call (never a low-confidence answer).

    `carried` supplies already-answered decisions from a previous run; those items are kept
    verbatim and only the unanswered ones are called again.
    """
    out: list[Decision] = []
    for i, item in enumerate(items, 1):
        if carried and i <= len(carried) and carried[i - 1].answered:
            out.append(carried[i - 1])
            continue
        state = Q.build_state(item["task"], item.get("context"))
        decision = backend.route(state, schema, timeout_s)
        attempt = 0
        while not decision.answered and attempt < retries:
            attempt += 1
            # Rate limits are the common failure for a hosted backend: back off far enough to
            # clear a per-minute window instead of hammering it three times in four seconds.
            time.sleep(max(sleep_s, min(20.0, 3.0 * (attempt ** 1.5))))
            decision = backend.route(state, schema, timeout_s)
        out.append(decision)
        if progress:
            mark = decision.tier if decision.answered else f"FAILED({(decision.error or '')[:40]})"
            print(f"  [{i:3d}/{len(items)}] {item['id']:26s} {mark}", file=sys.stderr, flush=True)
        if sleep_s:
            time.sleep(sleep_s)
    return out


def score(gold: list[dict[str, Any]], decisions: list[Decision], schema: dict[str, Any]) -> dict[str, Any]:
    """Per-question metrics for one backend."""
    tiers_gold = [it["labels"]["tier"] for it in gold]
    tiers_pred = [d.tier for d in decisions]
    probs = [d.tier_prob for d in decisions]
    latencies = [d.latency_ms for d in decisions]
    out: dict[str, Any] = {
        "tier": M.summarise_single(tiers_gold, tiers_pred, TIERS, probs, latencies),
        "failures": sum(1 for d in decisions if not d.answered),
        "failure_reasons": sorted({(d.error or "?")[:120] for d in decisions if not d.answered}),
    }
    tokens_in = sum(d.input_tokens or 0 for d in decisions)
    tokens_out = sum(d.output_tokens or 0 for d in decisions)
    out["tokens"] = {"input": tokens_in, "output": tokens_out}
    for flag in Q.FLAGS:
        flag_gold = [it["labels"][flag] for it in gold]
        flag_pred = [d.__getattribute__(flag) for d in decisions]
        # An unanswered item is a miss, not a silent pass: keep None in the vector so it scores
        # wrong rather than being dropped from the denominator.
        tp = sum(1 for g, p in zip(flag_gold, flag_pred) if g and p)
        fp = sum(1 for g, p in zip(flag_gold, flag_pred) if not g and p)
        fn = sum(1 for g, p in zip(flag_gold, flag_pred) if g and not p)
        evaluated = [p for p in flag_pred if p is not None]
        out[flag] = {
            "positive_rate_gold": sum(1 for g in flag_gold if g) / len(flag_gold),
            "positive_rate_pred": (sum(1 for p in evaluated if p) / len(evaluated)) if evaluated else None,
            "accuracy": sum(1 for g, p in zip(flag_gold, flag_pred) if p is not None and g == p) / len(flag_gold),
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
            "unanswered": len(flag_gold) - len(evaluated),
        }
    return out


def compare(gold: list[dict[str, Any]], decisions: dict[str, list[Decision]]) -> dict[str, Any]:
    """Pairwise comparison of the tier decision, with exact McNemar on the discordant items.

    Two variants per pair: over every item (a failed call scores as wrong, because that is what a
    caller would experience) and over the items both backends actually answered. When the two
    disagree, the difference is being driven by failures, not by routing quality — which is worth
    seeing rather than averaging away.
    """
    names = list(decisions)
    tiers_gold = [it["labels"]["tier"] for it in gold]
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_pred = [d.tier for d in decisions[a]]
            b_pred = [d.tier for d in decisions[b]]
            both_idx = [k for k in range(len(gold))
                        if decisions[a][k].answered and decisions[b][k].answered]
            entry = {"all_items": M.paired(tiers_gold, a_pred, b_pred, TIERS)}
            entry["both_answered"] = M.paired(
                [tiers_gold[k] for k in both_idx], [a_pred[k] for k in both_idx],
                [b_pred[k] for k in both_idx], TIERS)
            out[f"{a}_vs_{b}"] = entry
    return out


# --- reporting ---------------------------------------------------------------

def markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    names = list(result["metrics"])
    lines.append(f"corpus: {result['corpus']['items']} items (digest {result['corpus']['digest']}), "
                 f"schema {result['schema']['version']} (digest {result['schema']['digest']})")
    lines.append("")
    lines.append("| backend | tier acc | 95% CI | economy recall | frontier recall | p50 ms | p99 ms | tier ECE | failures |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for name in names:
        s = result["metrics"][name]["tier"]
        lat = s.get("latency_ms") or {}
        ece = s.get("ece")
        lo, hi = s["accuracy_ci95"]
        lines.append(
            f"| {name} | {s['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{_fmt(s['per_class_recall'].get('economy'))} | {_fmt(s['per_class_recall'].get('frontier'))} | "
            f"{_fmt(lat.get('p50'), 0)} | {_fmt(lat.get('p99'), 0)} | {_fmt(ece, 3)} | "
            f"{result['metrics'][name]['failures']} |"
        )
    lines.append("")
    lines.append("| backend | needs_tools acc | precision / recall | sensitive acc | precision / recall |")
    lines.append("|---|---:|---|---:|---|")
    for name in names:
        m = result["metrics"][name]
        lines.append(
            f"| {name} | {m['needs_tools']['accuracy']:.3f} | "
            f"{_fmt(m['needs_tools']['precision'])} / {_fmt(m['needs_tools']['recall'])} | "
            f"{m['sensitive']['accuracy']:.3f} | "
            f"{_fmt(m['sensitive']['precision'])} / {_fmt(m['sensitive']['recall'])} |"
        )
    for pair, variants in result["paired"].items():
        for variant, cmp in variants.items():
            lines.append("")
            lines.append(f"**{pair}** ({variant}) — both right {cmp['both_correct']}, "
                         f"first only {cmp['a_only_correct']}, second only {cmp['b_only_correct']}, "
                         f"neither {cmp['neither_correct']} (n={cmp['n']}); "
                         f"exact McNemar p = {cmp['mcnemar_exact_p']:.3f}")
    return "\n".join(lines)


def _fmt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


# --- cli ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure router backends on the labelled corpus.")
    ap.add_argument("--corpus", default=str(DATA / "steps.jsonl"))
    ap.add_argument("--schema", default=str(DATA / "questions.json"))
    ap.add_argument("--backends", default="laya,frontier",
                    help="comma-separated specs: laya | openai:<base_url>|<model>|<KEY_ENV>")
    ap.add_argument("--limit", type=int, default=0, help="score the first N items only")
    ap.add_argument("--out", default="", help="write the full result JSON here")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause between calls (rate limits)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--keep-raw", action="store_true", help="store each backend's raw reply")
    ap.add_argument("--env-file", default="", help="file of KEY=VALUE lines to load first "
                                                   "(e.g. your provider keys); values are never printed")
    ap.add_argument("--resume", default="", help="a previous results JSON: keep its answered "
                                                 "predictions and re-run only the unanswered items")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)

    schema = Q.load_schema(args.schema)
    items = load_corpus(args.corpus)
    if args.limit:
        items = items[:args.limit]

    backends: dict[str, Backend] = {}
    for spec in [s.strip() for s in args.backends.split(",") if s.strip()]:
        backend = parse_spec(spec)
        backends[backend.name] = backend
    if not backends:
        print("no backends requested", file=sys.stderr)
        return 2

    result: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "schema": {"path": str(args.schema), "version": schema["schema_version"],
                   "digest": Q.digest(schema), "questions": list(Q.questions_of(schema))},
        "corpus": {"path": str(args.corpus), "items": len(items), "digest": corpus_digest(items),
                   "tier_counts": {t: sum(1 for i in items if i["labels"]["tier"] == t) for t in TIERS},
                   "kinds": {k: sum(1 for i in items if i.get("kind") == k)
                             for k in sorted({i.get("kind", "task") for i in items})}},
        "backend_health": {name: b.health() for name, b in backends.items()},
        "predictions": {},
        "metrics": {},
    }

    decisions: dict[str, list[Decision]] = {}
    prior: dict[str, list[dict[str, Any]]] = {}
    if args.resume:
        previous = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        if previous["corpus"]["digest"] != corpus_digest(items):
            print("refusing to resume: the stored corpus digest differs from the current corpus",
                  file=sys.stderr)
            return 2
        prior = previous.get("predictions", {})
        result["resumed_from"] = str(args.resume)
    for name, backend in backends.items():
        if not args.quiet:
            print(f"# {name}: {backend.health()}", file=sys.stderr)
        carried = None
        if prior.get(name) and len(prior[name]) == len(items):
            carried = [decision_from_dict(d) for d in prior[name]]
            pending = sum(1 for d in carried if not d.answered)
            if not args.quiet:
                print(f"# {name}: resuming, {pending} unanswered item(s) to retry", file=sys.stderr)
        t0 = time.perf_counter()
        decisions[name] = run_backend(backend, items, schema, args.timeout, args.retries,
                                      args.sleep, progress=not args.quiet, carried=carried)
        if not args.quiet:
            print(f"# {name} done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        result["predictions"][name] = [d.to_dict(keep_raw=args.keep_raw) for d in decisions[name]]
        result["metrics"][name] = score(items, decisions[name], schema)

    result["paired"] = compare(items, decisions)
    result["report"] = markdown_report(result)
    print(result["report"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())