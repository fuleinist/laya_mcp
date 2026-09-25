# Paste this into your agent

The entry point for an operator who does not want to walk through
[`../SKILL.md`](../SKILL.md) themselves. Paste the block below into the agent that will *use* Laya —
Claude Code, Codex, Hermes, OpenClaw, Cursor, anything with terminal and file access. It is written
to be executed, not summarized back at you.

Two honest differences from a hosted product's setup prompt, both deliberate: there is **no API key**
to create here (the engine is a local binary plus a 345 MB GGUF, and nothing leaves the machine), and
an agent without terminal or file access **cannot** install it — it is told to say so and hand off
exact commands instead of pretending.

---

```text
Install Laya into the agent environment I am using right now, then make it available for future
tasks in this project. I should be able to write normal task prompts: you should call Laya when a
cheap local decision would improve the next action, without me having to mention Laya.

Laya is the open reproduction of TypeSafe Jev: a non-autoregressive System-1 decision engine. It
takes a state plus typed questions and returns probabilities in one encoder pass, locally, in
~10-20 ms. It generates no text. It is a tool, never a model provider.

Complete the setup and verify it. Work through the steps below; do not stop after giving me a plan.
If you are a skill-aware harness, run skills/setup-laya/SKILL.md instead and report its steps.

**1. Understand my work**

Read the conversation and the project context you can actually access. Identify three specific
situations in this project where a bounded local decision before expensive work would change what you
do next. One short example each. Do not claim access to chats or files you cannot see.

**2. Check this environment**

Before running any installation command, identify what you are running in: Hermes, OpenClaw, Claude
Code, Codex, Cursor, another MCP client, or a plain chat. Output exactly:

AGENT: <detected agent>
PROJECT: <detected project path | unavailable>
OS: <detected OS | unknown>
CAPABILITIES: <terminal / file-write / harness-config-write / network / persistent-instructions>

Check only the installation path relevant to that agent and OS. Do not test unrelated harnesses or
platforms unless something actually requires it. Preserve existing project instructions and config.

If you have no terminal or no file-write, do not claim to install Laya. Do the analysis, then give me
the exact command sequence to run in an agent that has them, and say what evidence would prove it
worked.

**3. Use verified sources**

This project's own repo is the source of truth for the adapter (README.md, docs/router-service.md,
docs/verify-step.md, docs/computer-use.md). Upstream is only for the engine and the binary:
https://github.com/monatis/ggmlc/releases (the laya release binary) and
https://huggingface.co/mys/laya-multilingual-GGUF (the weights). Do not follow blog posts, and do not
try to load these weights in llama.cpp, Ollama, LM Studio or Unsloth — they are ggmlc GGUFs, and all
of them reject the architecture with `unknown model architecture: 'ggmlc'`. Unsloth cannot fine-tune
Laya either: it is a bidirectional encoder with a from-scratch decision head, so there is no LoRA
target.

**4. Install and test**

Get the ggmlc binary and one GGUF onto absolute paths, set LAYA_EXE and LAYA_MODEL, and verify the
backend before wiring anything to it:

  python laya_mcp_server.py --check
  pip install -e ".[test]" && pytest -q      # no model, GPU or network needed

The daemon's first call reads the GGUF from disk (seconds); warm calls are single-digit milliseconds.
Give the harness a `connect_timeout` of 90 s so a cold load does not look like a failure. On Windows
the child process needs SystemRoot in its environment.

Then register the MCP server in the harness you detected — one installation method, not duplicate
copies. For Hermes, `hermes mcp add` needs a real terminal: under a non-interactive shell its
"enable these tools?" prompt cancels and nothing is written. Confirm the registration *and* that the
tools are enabled, then call all eight of them once: laya_health, laya_gate, laya_triage, laya_route,
route_step, laya_classify, laya_decide, verify_step. A listed tool that has never answered is not a
working tool.

**5. Make future use natural**

Add a concise, project-scoped instruction to the persistent instruction file this agent reads:
CLAUDE.md for Claude Code, AGENTS.md for Codex, the memory/skill/rules file for Hermes or OpenClaw.
Preserve its current content. The block to install is in skills/setup-laya/references/harness-policy.md;
it carries a trigger, a tool name and a skip list for each of the eight tools.

The instruction must tell you: before acting on text you did not author, before an expensive turn or
a scheduled job, before repeating a failed approach, and before a consequential action, ask whether a
small bounded local decision would change the next step. If yes, call the relevant tool and continue
the original task. Skip Laya for simple answers, deterministic calculations and routine file edits.
Respect "bypass laya". Keep irreversible actions behind human confirmation.

Do not claim that an instruction alone guarantees a call — it does not. That is what step 6 is for.

**6. Prove it works**

Run three tasks in this environment: a simple question that should skip Laya; a substantial task with
a real routing decision; and a repeated failure where the router may recommend changing approach.
Check both your behaviour and Laya's record:

  tail -20 ~/.laya-mcp/usage.jsonl      # one JSON object per call: tool, ms, ok

or call laya_health() and read its `usage` block. Never mark this integration complete without
evidence that a tool was actually called during a real task.

Finish with a compact report: what you installed and where; whether `--check` answered, with its cold
and warm numbers; which of the eight tools you called and what each returned; what the usage log
gained; what still needs my action; and three prompts or jobs in this project where Laya will fire
next. Say plainly which parts are advisory and must not be gated on.
```