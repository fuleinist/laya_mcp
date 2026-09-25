"""Tests for laya-router: schema, metrics, backends and the HTTP surface.

No model, no GPU and no network: the engine and the chat endpoint are both substituted. The live
end-to-end runs live in `laya_router/eval.py` and `probes/router_eval_probe.py`, which are run
deliberately and whose numbers are committed under `laya_router/data/results/`.
"""

import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from laya_router import metrics as M
from laya_router import questions as Q
from laya_router.backends import (Decision, LayaBackend, OpenAICompatBackend, parse_json_object,
                                  parse_spec)
from laya_router.eval import corpus_digest, load_corpus, markdown_report, run_backend, score
from laya_router.service import serve


# --- schema -----------------------------------------------------------------

def test_default_schema_loads_with_two_tiers():
    schema = Q.load_schema()
    assert schema["schema_version"].startswith("router-v")
    assert Q.tier_options(schema) == ["economy", "frontier"]
    assert set(Q.questions_of(schema)) == {"tier", "needs_tools", "sensitive"}


def test_schema_rejects_a_third_tier(tmp_path):
    """The base checkpoint's middle-tier recall is 0.13; a 3-tier router must fail loudly."""
    schema = json.loads(Q.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    schema["questions"]["tier"]["criteria"]["medium"] = "in between"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(Q.SchemaError, match="exactly two options"):
        Q.load_schema(path)


def test_schema_rejects_a_question_type_the_daemon_would_degrade(tmp_path):
    schema = json.loads(Q.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    schema["questions"]["needs_tools"]["type"] = "boolean"  # silently an empty choice daemon-side
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(Q.SchemaError, match="needs_tools"):
        Q.load_schema(path)


def test_schema_missing_file_is_explicit():
    with pytest.raises(Q.SchemaError, match="not found"):
        Q.load_schema("definitely/not/here.json")


def test_digest_is_stable_and_sensitive_to_wording():
    schema = Q.load_schema()
    same = Q.load_schema()
    assert Q.digest(schema) == Q.digest(same)
    edited = json.loads(json.dumps(schema))
    edited["questions"]["tier"]["criteria"]["economy"] = "different words entirely"
    assert Q.digest(edited) != Q.digest(schema), "a wording change must move the digest"


def test_build_state_requires_a_task_and_keeps_context_optional():
    assert Q.build_state("do the thing") == {"task": "do the thing"}
    assert Q.build_state("do the thing", "  some context ")["context"] == "some context"
    with pytest.raises(ValueError):
        Q.build_state("   ")


def test_render_prompt_carries_every_option_verbatim():
    schema = Q.load_schema()
    prompt = Q.render_prompt(schema, {"task": "cut a release"})
    for option, description in schema["questions"]["tier"]["criteria"].items():
        assert option in prompt and description in prompt
    assert "cut a release" in prompt


# --- metrics ----------------------------------------------------------------

def test_wilson_interval_matches_the_known_value():
    lo, hi = M.wilson(60, 100)
    assert (round(lo, 3), round(hi, 3)) == (0.502, 0.691)
    assert M.wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_is_one_for_a_symmetric_split():
    assert M.mcnemar_exact(38, 38) == 1.0


def test_mcnemar_exact_two_sided_matches_the_binomial_tail():
    # 10 vs 1 discordant pairs: p = 2 * sum_{k<=1} C(11,k) / 2^11
    expected = 2 * sum(math.comb(11, k) for k in range(2)) / 2 ** 11
    assert M.mcnemar_exact(10, 1) == pytest.approx(expected, rel=1e-12)
    assert M.mcnemar_exact(0, 0) == 1.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert M.ece([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == pytest.approx(0.0)
    assert M.ece([], []) is None


def test_per_class_recall_and_confusion():
    gold = ["economy", "economy", "frontier", "frontier"]
    pred = ["economy", "frontier", "economy", "frontier"]
    recalls = M.per_class_recall(gold, pred, ("economy", "frontier"))
    assert recalls == {"economy": 0.5, "frontier": 0.5}
    matrix = M.confusion(gold, pred, ("economy", "frontier"))
    assert matrix["economy"]["frontier"] == 1 and matrix["frontier"]["economy"] == 1


def test_paired_counts_only_the_discordant_items():
    gold = ["economy", "frontier", "economy", "frontier"]
    a = ["economy", "economy", "economy", "economy"]     # right on items 1 and 3
    b = ["frontier", "frontier", "economy", "frontier"]  # right on items 2, 3 and 4
    out = M.paired(gold, a, b, ("economy", "frontier"))
    assert (out["both_correct"], out["a_only_correct"], out["b_only_correct"]) == (1, 1, 2)
    assert out["neither_correct"] == 0 and out["disagreements_on_outcome"] == 3
    assert out["delta_accuracy"] == pytest.approx(-0.25)


def test_percentile_handles_small_samples():
    assert M.percentile([1.0], 0.5) == 1.0
    assert M.percentile([1.0, 3.0], 0.5) == pytest.approx(2.0)
    assert M.percentile([], 0.5) is None


# --- backends ---------------------------------------------------------------

def test_parse_json_object_strips_fences_and_preamble():
    assert parse_json_object('```json\n{"tier": "economy"}\n```')["tier"] == "economy"
    assert parse_json_object('Sure! {"tier": "frontier", "sensitive": true} done')["sensitive"] is True
    with pytest.raises(ValueError):
        parse_json_object("no object here")


def test_parse_spec_builds_a_laya_backend_by_default():
    assert isinstance(parse_spec("laya"), LayaBackend)
    assert isinstance(parse_spec(""), LayaBackend)
    backend = parse_spec("openai:http://localhost:8000/v1|some-model|SOME_KEY")
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "http://localhost:8000/v1" and backend.model == "some-model"
    assert backend.api_key_env == "SOME_KEY"
    with pytest.raises(ValueError, match="openai"):
        parse_spec("openai:just-a-url")


class FakeDaemon:
    """Stands in for the ggmlc daemon: same reply shape, no GPU."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def call(self, payload, timeout_ms=None):
        self.calls.append(payload)
        return self.reply


def _daemon_reply(tier="frontier", probs=(0.4661, 0.5339), tools=0.6173, sensitive=0.001):
    return {
        "answers": {
            "tier": {"type": "choice", "choice": tier, "confidence": 0.0033,
                     "probabilities": {"economy": probs[0], "frontier": probs[1]},
                     "action": {"act_probability": 1}},
            "needs_tools": {"type": "noul", "noul": tools, "confidence": 0.6},
            "sensitive": {"type": "noul", "noul": sensitive, "confidence": 0.999},
        },
        "usage": {"input_tokens": 194, "output_tokens": 0, "latency_ms": 12.5},
    }


def test_laya_backend_maps_the_daemon_reply_onto_the_decision():
    backend = LayaBackend(daemon=FakeDaemon(_daemon_reply()))
    schema = Q.load_schema()
    decision = backend.route(Q.build_state("cut a release"), schema)
    assert decision.answered and decision.tier == "frontier"
    assert decision.tier_prob == pytest.approx(0.5339), "calibration must use `probabilities`"
    assert decision.tier_confidence == pytest.approx(0.0033), "the daemon's `confidence` is kept too"
    assert decision.needs_tools is True and decision.sensitive is False
    assert decision.latency_ms == 12.5 and decision.input_tokens == 194


def test_laya_backend_sends_the_schema_questions_verbatim():
    daemon = FakeDaemon(_daemon_reply())
    schema = Q.load_schema()
    LayaBackend(daemon=daemon).route(Q.build_state("t"), schema)
    assert daemon.calls[0]["questions"] == Q.questions_of(schema)


def test_laya_backend_reports_a_failed_call_instead_of_guessing():
    class Broken(FakeDaemon):
        def call(self, payload, timeout_ms=None):
            raise RuntimeError("laya daemon stream closed")

    decision = LayaBackend(daemon=Broken({})).route(Q.build_state("t"), Q.load_schema())
    assert not decision.answered and decision.tier is None
    assert "stream closed" in decision.error


def test_laya_backend_flags_an_answer_with_no_tier_probabilities():
    reply = _daemon_reply()
    reply["answers"]["tier"] = {"type": "choice", "choice": "frontier", "probabilities": {}}
    decision = LayaBackend(daemon=FakeDaemon(reply)).route(Q.build_state("t"), Q.load_schema())
    assert not decision.answered and "no usable tier probability" in decision.error


# --- scoring and corpus -----------------------------------------------------

def _decision(tier, prob=0.8):
    return Decision(backend="x", tier=tier, tier_prob=prob, latency_ms=10.0,
                    needs_tools=True, needs_tools_prob=0.9, sensitive=False, sensitive_prob=0.1)


def test_score_counts_failures_as_misses_not_omissions():
    gold = [{"id": "a", "task": "t", "labels": {"tier": "economy", "needs_tools": True, "sensitive": False},
             "provenance": "x"},
            {"id": "b", "task": "t", "labels": {"tier": "frontier", "needs_tools": False, "sensitive": True},
             "provenance": "x"}]
    decisions = [_decision("economy"), Decision(backend="x", answered=False, tier=None, error="boom")]
    metrics = score(gold, decisions, Q.load_schema())
    assert metrics["tier"]["n"] == 2 and metrics["tier"]["correct"] == 1
    assert metrics["failures"] == 1
    assert metrics["needs_tools"]["accuracy"] == 0.5, "an unanswered item must not be dropped"


def test_compare_reports_the_answered_subset_separately():
    from laya_router.eval import compare

    gold = [{"id": "a", "task": "t", "labels": {"tier": "economy", "needs_tools": True, "sensitive": False},
             "provenance": "x"},
            {"id": "b", "task": "t", "labels": {"tier": "frontier", "needs_tools": True, "sensitive": False},
             "provenance": "x"}]
    decisions = {"laya": [_decision("economy"), _decision("frontier")],
                 "frontier": [_decision("frontier"),
                              Decision(backend="frontier", answered=False, tier=None, error="402")]}
    out = compare(gold, decisions)["laya_vs_frontier"]
    assert out["all_items"]["n"] == 2 and out["all_items"]["a_only_correct"] == 2, \
    "with every item counted, the failed call is a miss for that backend"
    assert out["both_answered"]["n"] == 1, "failures must be excluded from the subset variant"
    assert out["both_answered"]["a_only_correct"] == 1


def test_run_backend_reuses_carried_answers_and_only_retries_the_rest():
    calls = []

    class Counting(LayaBackend):
        def route(self, state, schema, timeout_s=60.0):
            calls.append(state["task"])
            return _decision("frontier")

    items = [{"id": "a", "task": "one", "labels": {"tier": "economy", "needs_tools": True, "sensitive": False},
              "provenance": "x"},
             {"id": "b", "task": "two", "labels": {"tier": "frontier", "needs_tools": True, "sensitive": False},
              "provenance": "x"}]
    carried = [_decision("economy"), Decision(backend="laya", answered=False, tier=None, error="cold")]
    out = run_backend(Counting(daemon=FakeDaemon({})), items, Q.load_schema(), 1.0, retries=0,
                      progress=False, carried=carried)
    assert calls == ["two"], "an answered item must not be called again"
    assert out[0].tier == "economy" and out[1].answered


def test_markdown_report_renders_both_tables():
    gold = [{"id": "a", "task": "t", "labels": {"tier": "economy", "needs_tools": True, "sensitive": False},
             "provenance": "x"}]
    decisions = {"laya": [_decision("economy")], "frontier": [_decision("frontier")]}
    result = {"corpus": {"items": 1, "digest": "deadbeef"},
                  "schema": {"version": "router-v1", "digest": "cafe"},
                  "metrics": {name: score(gold, d, Q.load_schema()) for name, d in decisions.items()},
                  "paired": {}}
    report = markdown_report(result)
    assert "| backend | tier acc" in report and "laya" in report and "frontier" in report


def test_committed_corpus_is_labelled_and_hashes_deterministically():
    items = load_corpus()
    assert len(items) >= 50, "the eval needs enough items for a paired comparison to mean anything"
    assert corpus_digest(items) == corpus_digest(load_corpus())
    tiers = {i["labels"]["tier"] for i in items}
    assert tiers == {"economy", "frontier"}, "both tiers must be represented"
    assert any(i["labels"]["sensitive"] for i in items)
    assert any(not i["labels"]["needs_tools"] for i in items), "reasoning-only steps must be present"
    assert all(i["provenance"] for i in items), "every item must name where it came from"


def test_corpus_rejects_an_unknown_tier_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "task": "t", "provenance": "p",
                                "labels": {"tier": "medium", "needs_tools": True, "sensitive": False}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="tier label"):
        load_corpus(path)


def test_run_backend_retries_a_failed_call(monkeypatch):
    class Flaky(LayaBackend):
        def __init__(self):
            super().__init__(daemon=FakeDaemon(_daemon_reply()))
            self.n = 0

        def route(self, state, schema, timeout_s=60.0):
            self.n += 1
            if self.n == 1:
                return Decision(backend="laya", answered=False, error="cold start")
            return _decision("frontier")

    monkeypatch.setattr("laya_router.eval.time.sleep", lambda *_: None)
    items = [{"id": "a", "task": "t", "labels": {"tier": "frontier", "needs_tools": True, "sensitive": False},
              "provenance": "x"}]
    out = run_backend(Flaky(), items, Q.load_schema(), 1.0, retries=1, progress=False)
    assert out[0].answered and out[0].tier == "frontier"


# --- the HTTP surface -------------------------------------------------------

class FakeBackend:
    name = "laya"  # registers under the default name, so `backend` may be omitted in requests

    def __init__(self, tier="economy", answered=True):
        self.tier = tier
        self.answered = answered
        self.seen = []

    def health(self):
        return {"name": self.name, "reachable": self.answered,
                **({} if self.answered else {"error": "engine down"})}

    def route(self, state, schema, timeout_s=60.0):
        self.seen.append(state)
        return Decision(backend=self.name, tier=self.tier if self.answered else None,
                        tier_prob=0.9, latency_ms=1.0, answered=self.answered,
                        error=None if self.answered else "engine down")


@pytest.fixture()
def live_server():
    backend = FakeBackend()
    httpd = serve(registry={backend.name: backend}, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, backend
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_reports_schema_and_backends(live_server):
    base, _ = live_server
    code, body = _get(f"{base}/health")
    assert code == 200 and body["ok"] is True
    assert body["service"] == "laya-router"
    assert body["schema_digest"] == Q.digest(Q.load_schema())
    assert body["backends"]["laya"]["reachable"] is True and body["advisory"] is True


def test_questions_serves_the_shape_both_backends_answer(live_server):
    base, _ = live_server
    code, body = _get(f"{base}/questions")
    assert code == 200
    assert body["tier_options"] == ["economy", "frontier"]
    assert set(body["questions"]) == {"tier", "needs_tools", "sensitive"}
    assert "<the step description>" in body["render_example"]


def test_route_over_get_and_post_agree(live_server):
    base, backend = live_server
    code_get, body_get = _get(f"{base}/route?task=cut+a+release")
    code_post, body_post = _post(f"{base}/route", {"task": "cut a release"})
    assert (code_get, code_post) == (200, 200)
    assert body_get["tier"] == body_post["tier"] == "economy"
    assert backend.seen[-1] == {"task": "cut a release"}
    assert body_post["advisory"] is True and "image-embedded" in body_post["boundary"]


def test_route_requires_a_task(live_server):
    base, _ = live_server
    code, body = _post(f"{base}/route", {"task": "  "})
    assert code == 400 and "task" in body["error"]
    code, body = _get(f"{base}/route")
    assert code == 400


def test_unknown_paths_and_backends_are_rejected(live_server):
    base, _ = live_server
    code, body = _get(f"{base}/nope")
    assert code == 404 and "no such path" in body["error"]
    code, body = _post(f"{base}/route", {"task": "t", "backend": "gpt5"})
    assert code == 400 and "unknown backend" in body["error"]


def test_a_failed_backend_call_is_a_502_not_a_decision(live_server):
    base, backend = live_server
    backend.answered = False
    code, body = _post(f"{base}/route", {"task": "cut a release"})
    assert code == 502 and body["answered"] is False and body["error"] == "engine down"


def test_health_is_unhealthy_when_no_backend_is_reachable():
    httpd = serve(registry={"laya": FakeBackend(answered=False)}, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/health", timeout=5):
            pass
    except urllib.error.HTTPError as exc:
        assert exc.code == 503
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)