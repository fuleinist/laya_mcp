"""Two backends behind one question schema: the local Laya daemon, and any chat endpoint.

Both answer the identical question object from `laya_router.questions`; both return the same
`Decision`. What differs is what they cost and how fast they answer, which is the whole point of
the comparison — so latency and token counts are measured here rather than estimated.

Declared backends (env-selectable, no new dependency; the chat client is stdlib `urllib`):

    laya                                        local ggmlc daemon (LAYA_EXE / LAYA_MODEL)
    openai:<base_url>|<model>|<KEY_ENV>         any OpenAI-compatible /chat/completions endpoint

`LAYA_ROUTER_FRONTIER` may hold the second form, so the service and the eval can name a frontier
backend without editing code.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from laya_router import questions as Q

FLAG_THRESHOLD = 0.5


def _flag(prob: float | None, threshold: float = FLAG_THRESHOLD) -> bool | None:
    return None if prob is None else prob >= threshold


@dataclass
class Decision:
    """One backend's answer to the shared schema, plus what it cost to get it."""

    backend: str
    tier: str | None = None
    tier_probabilities: dict[str, float] = field(default_factory=dict)
    tier_prob: float | None = None          # probability of the chosen tier (raw, uncalibrated)
    tier_confidence: float | None = None    # the daemon's own `confidence`; see note in docs
    needs_tools_prob: float | None = None
    needs_tools: bool | None = None
    sensitive_prob: float | None = None
    sensitive: bool | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    answered: bool = True                   # False: backend or parse failure, tier is not a guess
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, keep_raw: bool = False) -> dict[str, Any]:
        out = asdict(self)
        if not keep_raw:
            out.pop("raw", None)
        return out


class BackendError(RuntimeError):
    """A backend could not answer at all (as opposed to answering with a low probability)."""


class Backend:
    name = "backend"

    def route(self, state: dict[str, Any], schema: dict[str, Any], timeout_s: float = 60.0) -> Decision:
        raise NotImplementedError

    def answer(self, state: dict[str, Any], questions: dict[str, Any],
               timeout_s: float = 60.0, state_hint: str = "") -> Decision:
        """Answer arbitrary schema questions about `state` — step 3's `verify_step` path.

        `route()` is the routing specialisation; this is the general one, and both backends normalise
        into the daemon's answer shape (`{id: {"type": ..., "noul"|"choice": ...}}` in `Decision.raw`)
        so one parser reads either backend and a paired comparison compares answers, not parsers.
        """
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        raise NotImplementedError


class LayaBackend(Backend):
    """The local ggmlc daemon: `state` + `questions` in one encoder pass, typed answers out."""

    name = "laya"

    def __init__(self, daemon: Any | None = None) -> None:
        self._daemon = daemon

    @property
    def daemon(self) -> Any:
        if self._daemon is None:
            try:
                import laya_mcp_server as srv  # the sibling module in this repo
            except Exception as exc:  # pragma: no cover - import environment dependent
                raise BackendError(
                    f"laya backend needs laya_mcp_server on sys.path (run from the repo root): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._daemon = srv.DAEMON
        return self._daemon

    def health(self) -> dict[str, Any]:
        try:
            import laya_mcp_server as srv

            state = srv.DAEMON.health()
            return {"name": self.name, "reachable": True, **{k: state.get(k) for k in
                    ("exe", "model", "family", "device", "running")},
                    "schema": Q.digest(Q.load_schema())}
        except Exception as exc:
            return {"name": self.name, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    def route(self, state: dict[str, Any], schema: dict[str, Any], timeout_s: float = 60.0) -> Decision:
        payload = {"state": state, "questions": Q.questions_of(schema)}
        t0 = time.perf_counter()
        try:
            out = self.daemon.call(payload, timeout_ms=int(timeout_s * 1000))
        except Exception as exc:
            return Decision(backend=self.name, answered=False, latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(exc).__name__}: {exc}")
        latency = out.get("usage", {}).get("latency_ms")
        if latency is None:
            latency = (time.perf_counter() - t0) * 1000
        if out.get("error") and "answers" not in out:
            return Decision(backend=self.name, answered=False, latency_ms=latency,
                            error=str(out["error"]), raw=out)

        answers = out.get("answers", {})
        decision = Decision(backend=self.name, latency_ms=latency, raw=out,
                            input_tokens=out.get("usage", {}).get("input_tokens"),
                            output_tokens=out.get("usage", {}).get("output_tokens"))
        tier = answers.get(Q.TIER, {})
        if tier:
            probs = {k: float(v) for k, v in (tier.get("probabilities") or {}).items()}
            choice = tier.get("choice")
            decision.tier_probabilities = probs
            decision.tier = choice if choice in probs else (max(probs, key=probs.get) if probs else None)
            decision.tier_prob = probs.get(decision.tier) if decision.tier else None
            # For a two-option choice the daemon's `confidence` is not the option probability
            # (measured: 0.0033 reported while the options sat at 0.466/0.534). Keep both fields;
            # calibration must use `probabilities`.
            decision.tier_confidence = tier.get("confidence")
        for flag in Q.FLAGS:
            ans = answers.get(flag, {})
            if ans.get("type") == "noul":
                prob = float(ans.get("noul", 0.0))
                setattr(decision, f"{flag}_prob", prob)
                setattr(decision, flag, _flag(prob))
        if not decision.tier:
            decision.answered = False
            decision.error = "daemon returned no usable tier probability"
        return decision

    def answer(self, state: dict[str, Any], questions: dict[str, Any],
               timeout_s: float = 60.0, state_hint: str = "") -> Decision:
        """One daemon call with arbitrary questions; the daemon's answer shape passes through."""
        payload = {"state": state, "questions": questions}
        t0 = time.perf_counter()
        try:
            out = self.daemon.call(payload, timeout_ms=int(timeout_s * 1000))
        except Exception as exc:
            return Decision(backend=self.name, answered=False,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(exc).__name__}: {exc}")
        usage = out.get("usage") or {}
        latency = usage.get("latency_ms") or (time.perf_counter() - t0) * 1000
        if out.get("error") and "answers" not in out:
            return Decision(backend=self.name, answered=False, latency_ms=latency,
                            error=str(out["error"]), raw=out)
        answers = out.get("answers") or {}
        return Decision(backend=self.name, answered=bool(answers), latency_ms=latency, raw=out,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        error=None if answers else "daemon returned no answers")


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint answering the same schema as JSON."""

    name = "frontier"

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY",
                 name: str | None = None, key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 400) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self._key = key
        self.temperature = temperature
        self.max_tokens = max_tokens
        if name:
            self.name = name

    @property
    def key(self) -> str:
        if self._key is None:
            self._key = os.environ.get(self.api_key_env, "")
        return self._key

    def health(self) -> dict[str, Any]:
        return {"name": self.name, "reachable": bool(self.key), "base_url": self.base_url,
                "model": self.model, "key_env": self.api_key_env,
                **({} if self.key else {"error": f"{self.api_key_env} is not set"})}

    def _post(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def route(self, state: dict[str, Any], schema: dict[str, Any], timeout_s: float = 60.0) -> Decision:
        prompt = Q.render_prompt(schema, state)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You route steps. You answer with a single JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = time.perf_counter()
        try:
            out = self._post(body, timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            return Decision(backend=self.name, answered=False,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"HTTP {exc.code}: {detail}")
        except Exception as exc:
            return Decision(backend=self.name, answered=False,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000

        usage = out.get("usage") or {}
        decision = Decision(backend=self.name, latency_ms=latency, raw=out,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"))
        try:
            text = out["choices"][0]["message"]["content"]
        except Exception as exc:
            decision.answered = False
            decision.error = f"no message content in the reply: {exc}"
            return decision
        try:
            parsed = parse_json_object(text)
        except ValueError as exc:
            decision.answered = False
            decision.error = f"unparseable reply: {exc}"
            return decision

        tier = parsed.get(Q.TIER)
        options = Q.tier_options(schema)
        if isinstance(tier, str) and tier.strip().lower() in options:
            decision.tier = tier.strip().lower()
        else:
            decision.answered = False
            decision.error = f"tier {tier!r} is not one of {options}"
            return decision
        # A chat backend reports one confidence number, not a distribution: put it on both fields
        # so the two backends' calibration is computed from the same quantity.
        conf = parsed.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        decision.tier_confidence = conf
        decision.tier_prob = conf
        for flag in Q.FLAGS:
            value = parsed.get(flag)
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if isinstance(value, bool):
                setattr(decision, flag, value)
                setattr(decision, f"{flag}_prob", 1.0 if value else 0.0)
        return decision

    def answer(self, state: dict[str, Any], questions: dict[str, Any],
               timeout_s: float = 60.0, state_hint: str = "") -> Decision:
        """Answer arbitrary questions over a chat endpoint, normalised into the daemon's shape.

        A chat backend has no typed output channel: it is asked for one JSON object and its values
        are mapped onto the same answer space, question by question. A question the reply does not
        answer usefully is left out rather than guessed — `missing` in `raw` names it.
        """
        prompt = Q.render_answers_prompt(questions, state, state_hint)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You answer typed questions about a state. You reply with a single JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = time.perf_counter()
        try:
            out = self._post(body, timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            return Decision(backend=self.name, answered=False,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"HTTP {exc.code}: {detail}")
        except Exception as exc:
            return Decision(backend=self.name, answered=False,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            error=f"{type(exc).__name__}: {exc}")
        usage = out.get("usage") or {}
        decision = Decision(backend=self.name, latency_ms=(time.perf_counter() - t0) * 1000, raw=out,
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"))
        try:
            text = out["choices"][0]["message"]["content"]
        except Exception as exc:
            decision.answered = False
            decision.error = f"no message content in the reply: {exc}"
            return decision
        try:
            parsed = parse_json_object(text)
        except ValueError as exc:
            decision.answered = False
            decision.error = f"unparseable reply: {exc}"
            return decision

        answers, missing = normalise_answers(questions, parsed)
        decision.answered = bool(answers)
        decision.raw = {"answers": answers, "missing": missing, "text": text[:2000],
                        "usage": usage, "model": self.model}
        decision.error = None if not missing else f"no usable answer for {', '.join(missing)}"
        return decision


def normalise_answers(questions: dict[str, Any], parsed: dict[str, Any]
                      ) -> tuple[dict[str, Any], list[str]]:
    """Map a chat backend's JSON values onto the daemon's answer shape.

    Returns `(answers, missing)`: `answers` is `{id: {"type": ..., "noul"|"choice": ...}}`, identical
    to what the daemon produces, and `missing` lists the questions the reply did not answer in the
    question's own answer space. Loose reads are accepted (a `true` string, a differently-cased
    choice) because the mapping has to be *closed* to be safe, not strict.
    """
    answers: dict[str, Any] = {}
    missing: list[str] = []
    for name, q in questions.items():
        value = parsed.get(name)
        if q["type"] == "noul":
            if isinstance(value, bool):
                answers[name] = {"type": "noul", "noul": 1.0 if value else 0.0}
            elif isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no"):
                answers[name] = {"type": "noul",
                                 "noul": 1.0 if value.strip().lower() in ("true", "yes") else 0.0}
            else:
                missing.append(name)
        elif q["type"] == "choice":
            options = {str(k).strip().lower(): k for k in (q.get("criteria") or {})}
            key = str(value).strip().lower() if value is not None else ""
            if key in options:
                answers[name] = {"type": "choice", "choice": options[key]}
            else:
                missing.append(name)
        else:
            missing.append(name)
    return answers, missing


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a chat reply, fences and preamble included."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in {text[:120]!r}")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{exc}: {text[start:start + 160]!r}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def parse_spec(spec: str, name: str | None = None) -> Backend:
    """`laya` | `openai:<base_url>|<model>|<KEY_ENV>` | `openai:<base_url>|<model>`."""
    spec = (spec or "").strip()
    if spec in ("", "laya", "local"):
        return LayaBackend()
    if spec.startswith("openai:"):
        parts = spec.split(":", 1)[1].split("|")
        if len(parts) < 2:
            raise ValueError("openai backend spec is openai:<base_url>|<model>[|<KEY_ENV>]")
        base_url, model = parts[0], parts[1]
        key_env = parts[2] if len(parts) > 2 else "OPENAI_API_KEY"
        return OpenAICompatBackend(base_url=base_url, model=model, api_key_env=key_env, name=name)
    raise ValueError(f"unknown backend spec: {spec!r}")


def default_registry(frontier_spec: str | None = None) -> dict[str, Backend]:
    """The backends the service exposes: always `laya`; `frontier` when configured."""
    registry: dict[str, Backend] = {"laya": LayaBackend()}
    spec = frontier_spec or os.environ.get("LAYA_ROUTER_FRONTIER", "").strip()
    if spec:
        backend = parse_spec(spec, name="frontier")
        registry[backend.name] = backend
    return registry


# Kept for callers that want to inject a function backend (tests, replays).
FunctionBackend = Callable[[dict[str, Any], dict[str, Any], float], Decision]