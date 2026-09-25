# Agent skills

Procedure for an agent, in the `SKILL.md` format: YAML frontmatter with `name` and `description`,
then the steps the agent runs. This is deliberately a different artifact from `docs/` — the README
and the docs explain what the tools are and what was measured; a skill says what to run, in what
order, and what evidence makes the job done.

| harness | where a skill goes | how it is loaded |
|---|---|---|
| Claude Code | `~/.claude/skills/<name>/SKILL.md`, or `.claude/skills/` in a project | automatically, by description match |
| Hermes Agent | the profile's skills directory, one folder per skill | automatically; `skill_view(name)` reads it |
| OpenClaw / Codex | wherever its instruction file points | paste the body, or name the path in a task |
| Any MCP client with file access | anywhere on disk | name the path in the prompt |

An agent that does not read skills still has a path: hand it
[`setup-laya/references/operator-prompt.md`](setup-laya/references/operator-prompt.md), which is the
same six steps written as one paste-able prompt.

## `setup-laya`

[`setup-laya/SKILL.md`](setup-laya/SKILL.md) — install Laya and wire it into the detected harness:
check the ggmlc binary and a GGUF, verify with `--check`, register the MCP server, install the call
policy, then drive all eight tools and read the usage log as the acceptance test.

[`setup-laya/references/harness-policy.md`](setup-laya/references/harness-policy.md) — the text to
paste into the harness's persistent instruction file. A registered tool is not a used tool; this is
the part that decides whether the install does anything.

[`setup-laya/references/operator-prompt.md`](setup-laya/references/operator-prompt.md) — the
operator-facing entry point, for an agent that will not read `SKILL.md`.

`tests/test_skill.py` keeps these honest: the frontmatter has to parse, every tool named here has to
exist in `laya_mcp_server.py`, and every tool in the server has to be named here. A skill that drifts
from the tool surface is worse than no skill, because an agent follows it.