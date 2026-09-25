"""Are the two Laya backends' wire shapes the same? One formatter, two producers.

`laya_mcp_server._fmt` was written against the ggmlc `laya daemon` reply. Before adding an MLX
backend, the question that decides the integration cost is whether laya-mlx's `Agent.system_one()`
returns the same answer objects. This probe answers that by construction:

  1. offline — a fixture copied field-for-field from laya-mlx `laya_mlx/agent.py:258-285`
     (system_one) is pushed through `_fmt` and must round-trip unchanged;
  2. live — if LAYA_EXE + LAYA_MODEL are configured, one real daemon reply is captured and its
     answer objects are compared key-for-key with the fixture.

Run (live part optional):

    LAYA_EXE=/path/to/laya LAYA_MODEL=/path/to/laya_multilingual_q8_0.gguf \
      python mlx_schema_probe.py

Measured 2026-09-24 against ggmlc `laya` on Windows/CUDA CPU-device (multilingual Q8_0): both
producers emit identical answer objects. ggmlc adds `model`/`family`/`route`/`id` at the top level
and `latency_ms` inside `usage`; laya-mlx reports only `input_tokens`/`output_tokens`.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from laya_mcp_server import _fmt  # noqa: E402

# --- what laya-mlx returns for one call (agent.py:258-285, verbatim field set) ---------------
MLX_RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "department": {
            "type": "choice",
            "confidence": 0.9712,
            "action": {"act_probability": 0.8831},
            "choice": "billing",
            "probabilities": {"billing": 0.9712, "technical": 0.0181, "sales": 0.0107},
        },
        "urgency": {
            "type": "score",
            "confidence": 0.7418,
            "action": {"act_probability": 0.6205},
            "score": 1.2213,
            "legend": {"0": "not urgent", "1": "soon", "2": "critical"},
            "probabilities": {"0": 0.1902, "1": 0.5884, "2": 0.2214},
        },
        "refund": {
            "type": "noul",
            "confidence": 0.9604,
            "action": {"act_probability": 0.7710},
            "noul": 0.9604,
        },
    },
    "usage": {"input_tokens": 268, "output_tokens": 0},
}

# --- the ggmlc side, for the live comparison: same questions, same state --------------------
GGMLC_QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle the body?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs and outages", "sales": "pricing"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["not urgent", "soon", "critical"]},
    "refund": {"type": "noul", "instructions": "Does the sender ask for money back?"},
}
STATE = {"body": "I was charged twice for invoice 4411. Please refund the duplicate today."}
ERROR_BODY = {"error": "model not loaded"}

FAILURES = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46s} {detail}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    print("== offline: laya-mlx fixture through _fmt")
    out = json.loads(_fmt(MLX_RESULT))
    print(f"          summary: {out.get('summary', '(NONE)')}")
    check("department: billing (conf 0.971)" in out.get("summary", ""), "choice is summarised")
    check("refund: P(true)=0.960" in out.get("summary", ""), "noul is summarised")
    check("urgency: 1.221 on 3 levels" in out.get("summary", ""), "score+legend is summarised")
    for qid, answer in MLX_RESULT["answers"].items():
        check(json.dumps(out["answers"][qid], sort_keys=True) == json.dumps(answer, sort_keys=True),
              f"{qid} answer round-trips incl. action/probabilities")
    check(out["usage"] == MLX_RESULT["usage"], "usage round-trips")

    print("== offline: error body must pass through unchanged (ggmlc style)")
    check(json.loads(_fmt(ERROR_BODY)) == ERROR_BODY, "error passthrough")

    print("== live: real daemon reply, answer keys compared with the fixture")
    if not (os.environ.get("LAYA_EXE") and os.environ.get("LAYA_MODEL")):
        print("          skipped — set LAYA_EXE and LAYA_MODEL to run it")
    else:
        from laya_mcp_server import DAEMON

        try:
            live = DAEMON.call({"state": STATE, "questions": GGMLC_QUESTIONS})
        finally:
            DAEMON.stop()
        listed = live.get("answers", {})
        check(set(listed) == set(MLX_RESULT["answers"]), "same question ids answered",
              ", ".join(sorted(listed)))
        for qid, answer in listed.items():
            keys = set(answer) - {"probabilities"}
            expected = set(MLX_RESULT["answers"].get(qid, {})) - {"probabilities"}
            check(keys == expected, f"{qid} answer keys match laya-mlx", f"{sorted(keys)}")
        check("act_probability" in listed["refund"].get("action", {}),
              "action.act_probability is per-answer (as in laya-mlx)")
        check("latency_ms" in live.get("usage", {}),
              "ggmlc adds usage.latency_ms (absent from laya-mlx)", f"usage={live.get('usage')}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("OK: laya-mlx output is wire-compatible with the formatter; no adaption needed in _fmt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())