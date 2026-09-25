"""Accessibility-tree diffs, and the typed questions that verify a step from one.

Step 3 of the computer-use integration (issue #3). The pattern is the one `docs/computer-use.md`
§4.2 argues for: send the tree once, then only the deltas — a diff is ~200 tokens against a 12k-token
window, and it fits the engine's ~768-token state budget with room to spare. Two properties matter
and both are structural rather than calibrated:

* **Closed answer space.** Every question is `noul` (yes/no) or a two-or-three-way `choice`, so the
  screen text has no channel back to the planner as an instruction (the quarantined-perception
  pattern, §4.1).
* **The thing being asked about is in the question.** A diff is a shared state across questions, so
  an element spec that lived only in the state would be scored against every question at once.

Nothing here decides anything on its own: `verify_step` returns typed answers, and the *caller*
compares them with what it intended. That is deliberate — the engine has no notion of intent, and
`docs/router-service.md` measures how far its judgement goes.

This module is also the measurement harness: `build_questions()` generates items whose gold answer is
computable from the two trees, which is what makes step 3's accuracy a number rather than an
impression. Gold is never returned to the caller by `verify_step` — it exists for scoring only.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from laya_router import questions as Q
from laya_router.backends import Backend, Decision

DEFAULT_TEMPLATES = Path(__file__).resolve().parent / "data" / "questions.verify.json"
DEFAULT_CAPTURES = Path(os.environ.get("LAYA_A11Y_CAPTURES", Path.home() /
                                      "AppData/Local/hermes/cache/computer_use"))
# 40 lines each way is ~1,600 chars, ~730 state tokens on AX text at the measured 2.2 chars/token:
# the point of the exercise is a diff that fits the window, so the bound is part of the method.
MAX_DIFF_LINES = 40
DIALOG_ROLES = {"dialog", "window", "alert", "popup"}
ERROR_PATTERN = re.compile(r"\b(error|failed|failure|invalid|warning|cannot|denied|exception)\b", re.I)


# --- captures ----------------------------------------------------------------

class CaptureError(ValueError):
    """Raised for a capture that is not the shape cua-driver's `mode='ax'` produces."""


def parse_capture(data: Any) -> dict[str, Any]:
    """Accept a capture dict or its JSON text; return {app, window_title, elements}."""
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CaptureError(f"capture is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CaptureError(f"capture must be an object or its JSON text, got {type(data).__name__}")
    elements = data.get("elements")
    if not isinstance(elements, list) or not elements:
        raise CaptureError("capture has no non-empty `elements` list")
    clean = []
    for i, el in enumerate(elements):
        if not isinstance(el, dict):
            raise CaptureError(f"element {i} is not an object")
        clean.append({
            "role": str(el.get("role") or "").strip(),
            "label": " ".join(str(el.get("label") or "").split()),
            "bounds": el.get("bounds"),
        })
    return {"app": str(data.get("app") or ""), "window_title": str(data.get("window_title") or ""),
            "elements": clean}


def load_capture(path: str | os.PathLike[str]) -> dict[str, Any]:
    return parse_capture(Path(path).read_text(encoding="utf-8"))


def lines_of(capture: dict[str, Any], labelled_only: bool = True) -> list[str]:
    """Serialise a tree the way a diff will carry it: one `<role>: <label>` line per element."""
    out = []
    for el in capture["elements"]:
        if labelled_only and not el["label"]:
            continue
        out.append(f"{el['role'] or 'Element'}: {el['label']}")
    return out


def app_key(capture: dict[str, Any]) -> str:
    return capture["app"] or capture["window_title"] or "unknown"


# --- pairing real captures ---------------------------------------------------

def load_capture_dir(directory: str | os.PathLike[str] | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Every readable capture in a directory, oldest first."""
    directory = Path(directory) if directory else DEFAULT_CAPTURES
    if not directory.is_dir():
        raise CaptureError(f"no capture directory at {directory} "
                           "(set LAYA_A11Y_CAPTURES to the folder of cua-driver captures)")
    out = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            out.append((path, load_capture(path)))
        except (CaptureError, OSError, json.JSONDecodeError):
            continue  # a cache directory holds files that are not captures; skip, don't fail
    return out


def pair_captures(captures: Iterable[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Consecutive pairs of the *same* app: real before/after states of one window over time."""
    by_app: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, capture in captures:
        by_app.setdefault(app_key(capture), []).append((path, capture))
    pairs = []
    for app, items in sorted(by_app.items()):
        for (before_path, before), (after_path, after) in zip(items, items[1:]):
            pairs.append({"app": app, "before_path": str(before_path), "after_path": str(after_path),
                          "before": before, "after": after})
    return pairs


# --- diffs -------------------------------------------------------------------

def diff(before: dict[str, Any], after: dict[str, Any], max_lines: int = MAX_DIFF_LINES,
         variant: str = "labels") -> dict[str, Any]:
    """Added/removed lines between two trees, bounded so the result fits the state budget.

    The default serialisation is `labels`, which is not a style preference: on 103 real diffs it
    scored 0.602 against 0.515 for the `+ Role: label` form the docs originally sketched
    (`docs/verify-step.md`). Pass `variant` to re-measure.
    """
    before_lines, after_lines = lines_of(before), lines_of(after)
    before_counts, after_counts = _counter(before_lines), _counter(after_lines)
    added = [line for line in after_lines if before_counts.get(line, 0) < after_counts.get(line, 0)]
    removed = [line for line in before_lines if after_counts.get(line, 0) < before_counts.get(line, 0)]
    # Dedupe while keeping order: a repeated label in a tree is one line for this purpose.
    added = list(dict.fromkeys(added))
    removed = list(dict.fromkeys(removed))
    truncated = len(added) > max_lines or len(removed) > max_lines
    kept_added, kept_removed = added[:max_lines], removed[:max_lines]
    text = render_diff_text(kept_added, kept_removed, variant)
    return {"added": added, "removed": removed, "kept_added": kept_added,
            "kept_removed": kept_removed, "text": text, "truncated": truncated, "variant": variant,
            "lines": len(kept_added) + len(kept_removed), "chars": len(text),
            "approx_tokens": round(len(text) / 2.2),  # AX text measured at ~2.2 chars/token
            "elements_before": len(before["elements"]), "elements_after": len(after["elements"]),
            "line_delta": len(after_lines) - len(before_lines)}


DIFF_VARIANTS = ("prefix", "suffix", "prose", "labels")


def render_diff_text(kept_added: list[str], kept_removed: list[str],
                     variant: str = "prefix") -> str:
    """Serialise a diff, which is a choice this model is sensitive to.

    `computer-use.md` §3 measured serialisation swinging a head from 0.03 to 0.998 on identical
    content, so the same diff is offered four ways and the measurement decides which one works:

    * `prefix`  — `+ Role: label` / `- Role: label` (the default)
    * `suffix`  — `Role: label [appeared]` / `Role: label [gone]`
    * `prose`   — one sentence per side
    * `labels`  — labels only, no roles, with the +/- prefix
    """
    if variant not in DIFF_VARIANTS:
        raise ValueError(f"variant must be one of {DIFF_VARIANTS}, got {variant!r}")
    if variant == "prefix":
        return "\n".join([f"+ {l}" for l in kept_added] + [f"- {l}" for l in kept_removed])
    if variant == "suffix":
        return "\n".join([f"{l} [appeared]" for l in kept_added] +
                         [f"{l} [gone]" for l in kept_removed])
    if variant == "labels":
        return "\n".join([f"+ {role_and_label(l)[1]}" for l in kept_added] +
                         [f"- {role_and_label(l)[1]}" for l in kept_removed])
    appeared = "; ".join(role_and_label(l)[1] for l in kept_added)
    gone = "; ".join(role_and_label(l)[1] for l in kept_removed)
    return (f"Labels that appeared on screen: {appeared}\n"
            f"Labels that are gone from the screen: {gone}")


def _counter(lines: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return counts


def role_and_label(line: str) -> tuple[str, str]:
    role, _, label = line.partition(": ")
    return role.strip(), label.strip()


# --- questions ---------------------------------------------------------------

def load_templates(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    raw = json.loads(Path(path or DEFAULT_TEMPLATES).read_text(encoding="utf-8"))
    for name, spec in raw["questions"].items():
        if spec["type"] not in Q.QUESTION_TYPES:
            raise Q.SchemaError(f"verify template {name!r} has type {spec['type']!r}")
    return raw


def _informative(line: str, limit: int = 60) -> bool:
    """A line worth asking about: a real label, not a scaffold with an empty or numeric label."""
    _, label = role_and_label(line)
    if len(label) < 4 or len(label) > limit:
        return False
    return bool(re.search(r"[A-Za-z]{3}", label))


def _text_present_in(capture: dict[str, Any], role: str, label: str) -> bool:
    return any(el["role"] == role and el["label"] == label for el in capture["elements"])


def _any_label_match(capture: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    return any(pattern.search(el["label"]) for el in capture["elements"] if el["label"])


ROLE_ROTATION = ("TabItem", "MenuItem", "Hyperlink", "TreeItem", "ComboBox", "Edit")


def build_questions(pair: dict[str, Any], diff_result: dict[str, Any],
                    templates: dict[str, Any] | None = None,
                    pair_index: int = 0) -> list[dict[str, Any]]:
    """Concrete questions for one pair, each with the gold answer computed from the trees.

    Gold is derived, not authored: `present` asks about lines the diff actually added and removed
    (so the answer is checkable against the after-tree), `role_present` and `error_present` ask about
    a property of the current tree, and `net_added` counts the lines the model can see. Candidates
    come from the *rendered* lines, so a truncated diff never hides the evidence for a question.

    `role_present` rotates through a fixed role shortlist by pair — a corpus-level rule, applied
    before any answer is known — so the question covers roles with different base rates instead of
    one always-true role.
    """
    templates = templates or load_templates()
    spec = templates["questions"]
    after = pair["after"]
    items: list[dict[str, Any]] = []

    def add(kind: str, question: str, gold: Any, **extra: Any) -> None:
        items.append({"kind": kind, "name": f"{kind}_{len(items):02d}",
                      "type": spec[kind]["type"], "instructions": question, "gold": gold, **extra})

    for candidate in _pick(diff_result["kept_added"], 2):
        role, label = role_and_label(candidate)
        add("present", spec["present"]["instructions"].format(role=role, label=label),
            _text_present_in(after, role, label), role=role, label=label)
    for candidate in _pick(diff_result["kept_removed"], 2):
        role, label = role_and_label(candidate)
        add("present", spec["present"]["instructions"].format(role=role, label=label),
            _text_present_in(after, role, label), role=role, label=label)

    role = ROLE_ROTATION[pair_index % len(ROLE_ROTATION)]
    add("role_present", spec["role_present"]["instructions"].format(role=role),
        any((el["role"] or "") == role for el in after["elements"]), role=role)

    add("error_present", spec["error_present"]["instructions"],
        _any_label_match(after, ERROR_PATTERN))

    # Counted over the lines in the rendered diff, because that text is all the model sees — which
    # is also why a truncated diff makes this question answerable but not identical to the tree delta.
    added_n, removed_n = len(diff_result["kept_added"]), len(diff_result["kept_removed"])
    net = "appeared_more" if added_n > removed_n else ("removed_more" if removed_n > added_n else "equal")
    add("net_added", spec["net_added"]["instructions"], net, criteria=spec["net_added"]["criteria"])
    return items


def _pick(lines: list[str], n: int) -> list[str]:
    return [line for line in lines if _informative(line)][:n]


def to_laya_questions(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The `questions` object the daemon takes, one entry per item."""
    out: dict[str, Any] = {}
    for item in items:
        q: dict[str, Any] = {"type": item["type"], "instructions": item["instructions"]}
        if item["type"] == "choice":
            q["criteria"] = item["criteria"]
        out[item["name"]] = q
    return out


# --- answering ---------------------------------------------------------------

def answers_from(decision: Decision) -> dict[str, bool | str | None]:
    """Read the typed answers out of a daemon reply, keyed by question name."""
    out: dict[str, bool | str | None] = {}
    for name, answer in (decision.raw.get("answers") or {}).items():
        if answer.get("type") == "noul":
            out[name] = bool(answer.get("noul", 0) >= 0.5)
        elif answer.get("type") == "choice":
            out[name] = answer.get("choice")
    return out


def answer_details(decision: Decision) -> list[dict[str, Any]]:
    """Every typed answer with the probability behind it, so a caller can set its own threshold.

    A bare boolean hides how close the call was; the daemon reports a probability per question and
    this keeps it attached. (Calibration note from `docs/router-service.md`: for a two-option choice
    the daemon's `confidence` is not the option probability — read `probabilities`.)
    """
    out: list[dict[str, Any]] = []
    for name, answer in (decision.raw.get("answers") or {}).items():
        if answer.get("type") == "noul":
            prob = float(answer.get("noul", 0.0))
            out.append({"id": name, "type": "noul", "value": prob >= 0.5, "prob": prob})
        elif answer.get("type") == "choice":
            choice = answer.get("choice")
            probs = answer.get("probabilities") or {}
            out.append({"id": name, "type": "choice", "value": choice,
                        "prob": probs.get(choice) if choice else None,
                        "options": sorted(probs) or None})
    return out


def verify(backend: Backend, diff_result: dict[str, Any], items: list[dict[str, Any]],
           timeout_s: float = 60.0, templates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask one backend the generated questions about one diff. Typed answers out, nothing else.

    Both backends take the same `answer(state, questions)` path, so the comparison in `verify.py` is
    between two readers of one diff rather than between two integrations.
    """
    templates = templates or load_templates()
    questions = to_laya_questions(items)
    decision = backend.answer({"diff": diff_result["text"]}, questions, timeout_s,
                              state_hint=templates.get("state_hint", ""))
    return {"answered": decision.answered, "error": decision.error,
            "latency_ms": decision.latency_ms, "input_tokens": decision.input_tokens,
            "answers": answers_from(decision), "raw": decision.raw}