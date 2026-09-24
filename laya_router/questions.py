"""The shared question schema: load, validate, hash, and build the state both backends see.

A router decision is only comparable across backends if both were asked the *same* question.
This module is the single source of that question: `data/questions.json` is loaded, validated
against the daemon's three legal question types, and handed unchanged to the Laya backend (which
takes JSON questions verbatim) and rendered into the prompt of the OpenAI-compatible backend.
Rewording the tier options is a data edit, not a code edit — deliberately, because wording drove
21 accuracy points in the upstream study.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
DEFAULT_SCHEMA = Path(__file__).resolve().parent / "data" / "questions.json"
TIER = "tier"
FLAGS = ("needs_tools", "sensitive")


class SchemaError(ValueError):
    """Raised for a schema the daemon would silently degrade or misread."""


def load_schema(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read and validate a question schema file.

    Validation is client-side on purpose: the daemon degrades an unknown question type to an
    empty choice instead of erroring, so an unvalidated schema fails as a silent wrong answer.
    """
    path = Path(path) if path else DEFAULT_SCHEMA
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaError(f"question schema not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"question schema is not valid JSON: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SchemaError(f"question schema must be a JSON object, got {type(raw).__name__}")
    questions = raw.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("question schema needs a non-empty `questions` object")
    if TIER not in questions:
        raise SchemaError(f"question schema needs a {TIER!r} question: it is the routing decision")

    for name, q in questions.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question {name!r} must be an object")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise SchemaError(
                f"question {name!r} has type {qtype!r}; the daemon accepts only "
                f"{', '.join(QUESTION_TYPES)} and degrades anything else to an empty choice"
            )
        if qtype == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise SchemaError(f"choice question {name!r} needs at least two criteria")
            if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                raise SchemaError(f"choice question {name!r} needs a non-empty description per option")

    tier_criteria = questions[TIER]["criteria"]
    if len(tier_criteria) != 2:
        raise SchemaError(
            f"{TIER!r} must offer exactly two options (got {len(tier_criteria)}): the base "
            "checkpoint's middle-tier recall is 0.13, so a three-tier router is not supported by "
            "it — make the middle an escalation decision instead"
        )
    raw.setdefault("schema_version", "unversioned")
    return raw


def questions_of(schema: dict[str, Any]) -> dict[str, Any]:
    """The `questions` object, exactly as the daemon expects it."""
    return schema["questions"]


def digest(schema: dict[str, Any]) -> str:
    """Stable short hash of the schema — pinned into results so runs are comparable."""
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_state(task: str, context: str | None = None) -> dict[str, Any]:
    """The state both backends are scored against.

    One state, one question set: Laya scores a single state per call, so anything that must
    distinguish two candidates has to live in the question text, not in extra state fields.
    """
    task = (task or "").strip()
    if not task:
        raise ValueError("task must be a non-empty string")
    state: dict[str, Any] = {"task": task}
    if context and context.strip():
        state["context"] = context.strip()
    return state


def tier_options(schema: dict[str, Any]) -> list[str]:
    return list(schema["questions"][TIER]["criteria"])


def render_prompt(schema: dict[str, Any], state: dict[str, Any]) -> str:
    """The same schema, as text, for a chat backend.

    Both backends must answer the same question; this rendering is the *only* place the question
    is restated for the chat backend, and it is generated from the identical JSON object.
    """
    lines = [
        "You are routing one step of an agent's work. Answer the questions below about the step.",
        f"Step description: {state.get('task', '')}",
    ]
    if state.get("context"):
        lines.append(f"Context: {state['context']}")
    lines.append("")
    lines.append("Questions:")
    for name, q in questions_of(schema).items():
        lines.append(f"- {name} ({q['type']}): {q.get('instructions', '')}")
        for option, description in (q.get("criteria") or {}).items():
            lines.append(f"    - {option}: {description}")
    lines.append("")
    lines.append(
        "Reply with ONE JSON object and nothing else, with these keys: "
        + ", ".join(f'"{name}"' for name in questions_of(schema))
        + ('. "tier" is one of: ' + ", ".join(f'"{o}"' for o in tier_options(schema)) if TIER in questions_of(schema) else "")
        + '. The noul questions take true or false. Add "confidence" as a number in [0,1] for your tier answer.'
    )
    return "\n".join(lines)