"""Step 3: accessibility diffs, the typed questions, and the verify_step harness.

The synthetic trees below are shaped exactly like a cua-driver `mode='ax'` capture, so the gold
answers are checkable by hand and the tests never need an engine, a screen, or the network. The
scoring tests use handcrafted rows for the same reason: a metric that can only be checked by running
a model is not a metric anybody can review.
"""

from __future__ import annotations

import json

import pytest

from laya_router import a11y, verify
from laya_router.backends import Decision


def capture(app: str, elements: list[tuple[str, str]]) -> dict:
    return a11y.parse_capture({"app": app, "window_title": f"{app} window",
                               "elements": [{"role": r, "label": l, "bounds": [0, 0, 4, 4]}
                                            for r, l in elements]})


BEFORE = capture("test.exe", [
    ("Button", "Refresh"), ("Text", "Ready"), ("Button", "Apply filters"),
    ("MenuItem", "Export"), ("Text", "3 items"),
])
AFTER = capture("test.exe", [
    ("Button", "Refresh"), ("Text", "Ready"), ("Button", "Apply filters"),
    ("Text", "Error while saving the file"), ("MenuItem", "Export"),
])


# --- captures ----------------------------------------------------------------

def test_parse_capture_accepts_a_dict_or_its_json_text():
    as_dict = a11y.parse_capture({"app": "a.exe", "elements": [{"role": "Button", "label": "Go"}]})
    as_text = a11y.parse_capture(json.dumps({"app": "a.exe",
                                             "elements": [{"role": "Button", "label": "Go"}]}))
    assert as_dict["elements"] == as_text["elements"] == [{"role": "Button", "label": "Go",
                                                           "bounds": None}]
    assert as_dict["app"] == "a.exe" and as_text["app"] == "a.exe"


def test_parse_capture_normalises_whitespace_in_labels():
    cap = a11y.parse_capture({"elements": [{"role": "Text", "label": "  two\n  words  "}]})
    assert cap["elements"][0]["label"] == "two words"


@pytest.mark.parametrize("bad", [
    {"elements": []},
    {"elements": "not a list"},
    {"elements": ["not an object"]},
    {"no_elements": True},
    "[1, 2, 3]",
])
def test_parse_capture_refuses_what_is_not_a_capture(bad):
    with pytest.raises(a11y.CaptureError):
        a11y.parse_capture(bad)


def test_lines_of_serialises_role_and_label_and_can_skip_unlabelled():
    cap = capture("a.exe", [("Button", "Go"), ("Group", ""), ("Text", "Hi")])
    assert a11y.lines_of(cap) == ["Button: Go", "Text: Hi"]
    assert len(a11y.lines_of(cap, labelled_only=False)) == 3


def test_load_capture_dir_skips_files_that_are_not_captures(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps({"app": "x", "elements": [{"role": "Text",
                                                                              "label": "hi"}]}))
    (tmp_path / "junk.json").write_text("{not json")
    (tmp_path / "notes.md").write_text("ignore me")
    found = a11y.load_capture_dir(tmp_path)
    assert [p.name for p, _ in found] == ["good.json"]


def test_load_capture_dir_names_the_setting_when_the_directory_is_missing(tmp_path):
    with pytest.raises(a11y.CaptureError) as exc:
        a11y.load_capture_dir(tmp_path / "nope")
    assert "LAYA_A11Y_CAPTURES" in str(exc.value)


# --- pairing -----------------------------------------------------------------

def test_pair_captures_pairs_consecutive_captures_of_one_app(tmp_path):
    paths = []
    for i, app in enumerate(["a.exe", "a.exe", "b.exe", "a.exe"]):
        p = tmp_path / f"c{i}.json"
        p.write_text(json.dumps({"app": app, "elements": [{"role": "Text", "label": f"n{i}"}]}))
        paths.append(p)
    pairs = a11y.pair_captures(a11y.load_capture_dir(tmp_path))
    assert [(p["app"], p["before"]["elements"][0]["label"], p["after"]["elements"][0]["label"])
            for p in pairs] == [("a.exe", "n0", "n1"), ("a.exe", "n1", "n3")]


def test_pair_captures_needs_two_captures_of_the_same_app():
    only = [("p1", capture("a.exe", [("Text", "hi")])), ("p2", capture("b.exe", [("Text", "hi")]))]
    assert a11y.pair_captures(only) == []


# --- diffs -------------------------------------------------------------------

def test_diff_separates_what_appeared_from_what_went():
    d = a11y.diff(BEFORE, AFTER)
    assert d["added"] == ["Text: Error while saving the file"]
    assert d["removed"] == ["Text: 3 items"]
    assert d["variant"] == "labels"
    assert d["elements_before"] == 5 and d["elements_after"] == 5 and d["line_delta"] == 0


def test_diff_does_not_crash_on_a_label_that_exists_only_in_after():
    # Regression: an unguarded counter lookup raised KeyError on exactly this shape.
    d = a11y.diff(capture("a.exe", [("Text", "one")]), capture("a.exe", [("Text", "two")]))
    assert d["added"] == ["Text: two"] and d["removed"] == ["Text: one"]


def test_diff_counts_a_repeated_label_by_its_multiplicity():
    before = capture("a.exe", [("Text", "item"), ("Text", "item")])
    after = capture("a.exe", [("Text", "item"), ("Text", "item"), ("Text", "item")])
    assert a11y.diff(before, after)["added"] == ["Text: item"]


def test_diff_bounds_itself_and_says_so():
    before = capture("a.exe", [("Text", f"old {i}") for i in range(60)])
    after = capture("a.exe", [("Text", f"new {i}") for i in range(60)])
    d = a11y.diff(before, after, max_lines=5)
    assert d["truncated"] is True and d["lines"] == 10
    assert len(d["kept_added"]) == 5 and len(d["added"]) == 60
    assert d["text"].count("+ ") == 5


@pytest.mark.parametrize("variant,marker", [
    ("prefix", "+ Text: Error while saving the file"),
    ("suffix", "Text: Error while saving the file [appeared]"),
    ("labels", "+ Error while saving the file"),
    ("prose", "Labels that appeared on screen: Error while saving the file"),
])
def test_diff_variants_carry_the_same_lines_differently(variant, marker):
    d = a11y.diff(BEFORE, AFTER, variant=variant)
    assert marker in d["text"]
    assert "3 items" in d["text"]


def test_render_diff_text_refuses_an_unknown_variant():
    with pytest.raises(ValueError):
        a11y.render_diff_text(["Text: a"], [], "yaml")


# --- questions ---------------------------------------------------------------

def test_things_asked_about_are_in_the_question_not_the_state():
    items = a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER))
    by_kind: dict[str, list[dict]] = {}
    for item in items:
        by_kind.setdefault(item["kind"], []).append(item)
    # An element spec in the shared state would be scored against every question at once.
    for item in by_kind["present"]:
        assert item["label"] in item["instructions"] and item["role"] in item["instructions"]
    assert len(by_kind["present"]) == 2  # the one line added and the one line removed
    assert len(by_kind["net_added"]) == 1  # counted over the rendered lines


def test_present_gold_is_read_from_the_tree_not_assumed():
    items = [i for i in a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER))
             if i["kind"] == "present"]
    golds = {(i["label"], i["gold"]) for i in items}
    assert ("Error while saving the file", True) in golds   # it is in the after-tree
    assert ("3 items", False) in golds                            # it is gone
    for item in items:
        assert item["gold"] == a11y._text_present_in(AFTER, item["role"], item["label"])


def test_net_added_gold_counts_only_the_lines_in_the_rendered_diff():
    before = capture("a.exe", [("Text", f"old {i}") for i in range(20)])
    after = capture("a.exe", [("Text", f"new {i}") for i in range(20)])
    d = a11y.diff(before, after, max_lines=4)
    item = [i for i in a11y.build_questions({"after": after}, d, pair_index=0)
            if i["kind"] == "net_added"][0]
    assert item["gold"] == "equal"  # 4 kept each way, even though the trees differ by 20 each way


def test_role_present_rotates_through_the_shortlist():
    roles = []
    for i in range(len(a11y.ROLE_ROTATION)):
        item = [x for x in a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER),
                                                pair_index=i) if x["kind"] == "role_present"][0]
        roles.append(item["role"])
    assert roles == list(a11y.ROLE_ROTATION)


def test_role_present_gold_matches_the_tree():
    item = [x for x in a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER),
                                            pair_index=1) if x["kind"] == "role_present"][0]
    assert item["role"] == "MenuItem"
    assert item["gold"] is True  # AFTER keeps its MenuItem


def test_error_present_answers_from_the_after_tree():
    with_error = [x for x in a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER))
                  if x["kind"] == "error_present"][0]
    without = [x for x in a11y.build_questions({"after": BEFORE}, a11y.diff(BEFORE, AFTER))
               if x["kind"] == "error_present"][0]
    assert with_error["gold"] is True and without["gold"] is False


def test_to_laya_questions_gives_the_daemon_one_entry_per_item():
    items = a11y.build_questions({"after": AFTER}, a11y.diff(BEFORE, AFTER))
    questions = a11y.to_laya_questions(items)
    assert set(questions) == {i["name"] for i in items}
    for name, q in questions.items():
        assert q["type"] in ("noul", "choice") and q["instructions"]
        assert ("criteria" in q) == (q["type"] == "choice")


def test_load_templates_refuses_a_question_type_the_daemon_would_degrade(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "x",
                               "questions": {"q": {"type": "essay", "instructions": "write"}}}))
    with pytest.raises(Exception):
        a11y.load_templates(bad)


# --- reading the answers -----------------------------------------------------

def decision(answers: dict) -> Decision:
    return Decision(backend="fake", answered=True, latency_ms=7.0, input_tokens=101,
                    raw={"answers": answers, "usage": {"latency_ms": 7.0, "input_tokens": 101}})


def test_answers_from_reads_noul_by_threshold_and_choice_by_value():
    out = a11y.answers_from(decision({
        "present_00": {"type": "noul", "noul": 0.71},
        "present_01": {"type": "noul", "noul": 0.29},
        "net_added_05": {"type": "choice", "choice": "equal", "probabilities":
                         {"equal": 0.5, "appeared_more": 0.3, "removed_more": 0.2}},
    }))
    assert out == {"present_00": True, "present_01": False, "net_added_05": "equal"}


def test_answer_details_keeps_the_probability_so_a_caller_can_threshold():
    details = a11y.answer_details(decision({
        "present_00": {"type": "noul", "noul": 0.71},
        "net_added_05": {"type": "choice", "choice": "equal",
                         "probabilities": {"equal": 0.5, "removed_more": 0.5}},
    }))
    by_id = {d["id"]: d for d in details}
    assert by_id["present_00"]["value"] is True and by_id["present_00"]["prob"] == 0.71
    assert by_id["net_added_05"]["prob"] == 0.5 and by_id["net_added_05"]["options"] == [
        "equal", "removed_more"]


class FakeAnswerer:
    """A backend that answers the questions it is given, so `verify()` can be tested offline."""

    name = "fake"

    def __init__(self, choices: dict[str, object] | None = None):
        self.choices = choices or {}
        self.seen: list[tuple[dict, dict, str]] = []

    def answer(self, state, questions, timeout_s=60.0, state_hint=""):
        self.seen.append((state, questions, state_hint))
        answers = {}
        for name, q in questions.items():
            if name in self.choices:
                value = self.choices[name]
            else:
                value = True if q["type"] == "noul" else next(iter(q.get("criteria") or {}))
            answers[name] = ({"type": "noul", "noul": 1.0 if value else 0.0} if q["type"] == "noul"
                             else {"type": "choice", "choice": value})
        return decision(answers)


def test_verify_sends_the_rendered_diff_and_the_question_text():
    backend = FakeAnswerer()
    d = a11y.diff(BEFORE, AFTER)
    items = a11y.build_questions({"after": AFTER}, d)
    result = a11y.verify(backend, d, items, 5.0)
    state, questions, hint = backend.seen[0]
    assert state == {"diff": d["text"]}
    assert set(questions) == {i["name"] for i in items}
    assert items[0]["label"] in questions[items[0]["name"]]["instructions"]
    assert hint and result["answered"] is True and result["input_tokens"] == 101


# --- the harness -------------------------------------------------------------

def test_build_items_caps_a_repaint_to_the_character_budget():
    before = capture("a.exe", [("Text", f"old {i}") for i in range(200)])
    after = capture("a.exe", [("Text", f"new {i}") for i in range(200)])
    pair = {"app": "a.exe", "before": before, "after": after, "before_path": "b", "after_path": "a"}
    work = verify.build_items([pair], max_chars=400)
    entry = work[0]
    assert entry["diff"]["chars"] <= 400 and entry["diff"]["truncated"] is True
    assert entry["diff"]["text"] and entry["items"]


def row(kind: str, gold, pred, pair: int = 0, tokens: int = 100, latency: float = 5.0) -> dict:
    return {"pair": pair, "app": "a.exe", "kind": kind, "name": f"{kind}_{pair}",
            "type": "noul" if isinstance(gold, bool) else "choice", "role": None,
            "gold": gold, "pred": pred, "correct": pred == gold, "answered": True,
            "latency_ms": latency, "input_tokens": tokens, "diff_chars": 120, "diff_lines": 4,
            "diff_truncated": False, "variant": "labels", "error": None}


def test_score_reports_the_majority_baseline_next_to_accuracy():
    rows = [row("present", True, True, pair=0), row("present", True, False, pair=1),
            row("present", True, True, pair=2), row("present", False, True, pair=3),
            row("count", 1, 1, pair=0), row("count", 2, 1, pair=1)]
    scored = verify.score(rows)
    assert scored["overall"]["n"] == 6 and scored["overall"]["correct"] == 3
    present = scored["by_kind"]["present"]
    assert present["n"] == 4 and present["correct"] == 2
    assert present["majority_baseline"] == 0.75      # three of four golds are True
    assert present["gold_counts"] == {"True": 3, "False": 1}
    assert scored["by_kind"]["count"]["majority_baseline"] == 0.5
    assert scored["overall"]["accuracy_ci95"][0] < 0.5 < scored["overall"]["accuracy_ci95"][1]


def test_score_counts_an_unanswered_item_as_a_miss_not_a_quiet_pass():
    rows = [row("present", True, None, pair=0), row("present", True, True, pair=1)]
    scored = verify.score(rows)
    assert scored["overall"]["correct"] == 1 and scored["overall"]["unanswered"] == 1
    assert scored["failures"] == 0  # the call answered; this one item did not


def test_score_surfaces_the_token_profile_against_the_state_budget():
    rows = [row("present", True, True, pair=0, tokens=500),
            row("present", True, True, pair=1, tokens=2000)]
    size = verify.score(rows)["size"]
    assert size["engine_input_tokens"]["max"] == 2000
    assert size["state_budget_tokens"] == 768
    assert size["over_budget_share"] == 0.5
    assert size["truncated_share"] == 0.0


def test_compare_scores_only_the_items_both_backends_answered():
    a = [row("present", True, True, pair=0), row("present", False, False, pair=1)]
    b = [row("present", True, True, pair=0), row("present", False, True, pair=1)]
    out = verify.compare(a, b, "laya", "frontier")
    assert out["n"] == 2 and out["both_correct"] == 1
    assert out["a_only_correct"] == 1 and out["b_only_correct"] == 0
    assert out["a"] == "laya" and out["scope"].startswith("items asked of both")


def test_render_markdown_reports_kinds_baselines_sizes_and_the_pairing():
    result = {
        "schema": "verify-v1", "variant": "labels", "captures": 2, "pairs": 1,
        "capture_dir": "somewhere",
        "metrics": {"laya": verify.score([row("present", True, True)])},
        "paired": {"a": "laya", "b": "frontier", "scope": "items both answered",
                   "both_correct": 1, "a_only_correct": 0, "b_only_correct": 0,
                   "neither_correct": 0, "mcnemar_exact_p": 1.0},
    }
    text = verify.render_markdown(result)
    assert "majority baseline" in text and "present" in text
    assert "state budget" in text or "budget" in text
    assert "McNemar exact p = 1.000" in text