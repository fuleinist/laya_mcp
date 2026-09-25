"""Tests for the agent skills in `skills/`.

A skill is an instruction an agent follows, so a skill that has drifted from the tool surface is
worse than no skill at all: the agent calls a tool that does not exist, or stops calling one that
does. These tests are the drift guard, and they need no model, GPU, network or `mcp` package — the
tool list is read from the server's source, which is the same `@logged(...)` name the usage log
records.

    pytest tests/test_skill.py -q
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
POLICY = SKILLS / "setup-laya" / "references" / "harness-policy.md"
OPERATOR_PROMPT = SKILLS / "setup-laya" / "references" / "operator-prompt.md"
SERVER = ROOT / "laya_mcp_server.py"

# The decorator sits under @mcp.tool and is what the usage log records, so it names the tool that
# actually ran — not the function name (route_step and verify_step would not be found by the latter).
LOGGED = re.compile(r'@logged\(\s*"([a-z][a-z0-9_]*)"')
# A tool-shaped call in prose or in the paste-able block: `laya_gate(text)`, `route_step(task)`.
CALLED = re.compile(r"\b((?:laya_|route_|verify_)[a-z0-9_]+)\s*\(")


def _server_tools() -> set:
    return set(LOGGED.findall(SERVER.read_text(encoding="utf-8")))


def _skill_files() -> list:
    return sorted(p for p in SKILLS.rglob("*.md"))


def _all_skill_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _skill_files())


def _fenced(text: str, language: str, index: int = 0) -> str:
    """The body of the index-th ```<language> fence."""
    blocks = re.findall(r"```" + language + r"\n(.*?)```", text, re.S)
    assert len(blocks) > index, f"no ```{language} block at index {index}"
    return blocks[index]


def test_the_server_still_registers_eight_tools():
    # A guard on the guard: if this count changes, every test below is measuring the wrong surface,
    # and the README's "eight tools" claim needs revisiting in the same change.
    assert len(_server_tools()) == 8


def test_every_skill_has_frontmatter_an_agent_can_select_on():
    found = sorted(SKILLS.glob("*/SKILL.md"))
    assert found, "no skills/*/SKILL.md"
    for path in found:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{path.name}: frontmatter must be the first thing in the file"
        _, frontmatter, _ = text.split("---\n", 2)
        fields = dict(
            line.split(":", 1) for line in frontmatter.strip().splitlines() if ":" in line
        )
        name = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        assert name == path.parent.name, f"{path.name}: frontmatter name {name!r} != directory name"
        assert description, f"{path.name}: empty description"
        assert len(description) <= 1024, f"{path.name}: description is {len(description)} chars"
        # Selection happens on the description alone, so it has to open with its trigger.
        assert description.startswith("Use when"), f"{path.name}: description must open with 'Use when'"


def test_the_skills_invent_no_tools():
    known = _server_tools()
    called = {name for name in CALLED.findall(_all_skill_text())}
    unknown = sorted(called - known)
    assert not unknown, f"skills call tools the server does not register: {unknown}"


def test_the_setup_skill_names_every_registered_tool():
    text = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((SKILLS / "setup-laya").rglob("*.md"))
    )
    missing = sorted(name for name in _server_tools() if not re.search(rf"\b{name}\b", text))
    assert not missing, f"setup-laya never names these registered tools: {missing}"


def test_the_pasteable_policy_lists_exactly_the_registered_tools():
    # The block in section 1 is copied verbatim into an operator's instruction file, so its call list
    # is the artifact that has to match the surface — one bullet per tool, no bullet for a ghost.
    block = _fenced(POLICY.read_text(encoding="utf-8"), "text")
    listed = set(re.findall(r"^\s*-\s+([a-z][a-z0-9_]*)\s*\(", block, re.M))
    assert listed == _server_tools(), (
        f"policy lists {sorted(listed)}; server registers {sorted(_server_tools())}"
    )
    for phrase in ("Skip it for:", "bypass laya"):
        assert phrase in block, f"the policy block lost its {phrase!r} escape hatch"


def test_the_operator_prompt_carries_no_hosted_product_ceremony():
    # This file was written against a hosted setup prompt, whose shape is an API key and a console
    # URL. Laya is a local binary: inheriting that ceremony would send an operator looking for a key
    # that does not exist.
    block = _fenced(OPERATOR_PROMPT.read_text(encoding="utf-8"), "text")
    for bad in ("api key", "api_key", "console.", "sign up", "signup"):
        assert bad not in block.lower(), f"operator prompt asks for {bad!r}; Laya is local"
    assert "usage.jsonl" in block, "operator prompt must name the evidence it will be judged on"


def test_every_relative_link_in_the_skills_resolves():
    broken = []
    for path in _skill_files():
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, f"broken relative links: {broken}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))