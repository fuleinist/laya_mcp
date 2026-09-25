# Setting up a computer-use API with Laya as the System-1 router

Source: *Smart routing with an open source System 1 model — Laya* (Gleb Lukicov, 2026-09-20) and its
repo [`glukicov/laya_router`](https://github.com/glukicov/laya_router) (`docs/EVAL.md`, `results/`).

## 0. What the study actually establishes

| | route accuracy | needs_tools | is_sensitive | ECE | p50 | p99 | per 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Laya 421M, local (M4 MPS, FP32, 512 ctx) | 0.600 | **0.867** | 0.733 | **0.093** | **184 ms** | **233 ms** | **$0.00** |
| GPT-5 nano (structured outputs) | 0.600 | 0.711 | **0.894** | 0.172 | 6,415 ms | 14,751 ms | $0.58 |

Three findings transfer directly to a computer-use design, and each one is a constraint:

1. **A tie on accuracy, a 35× latency gap, zero token cost.** They disagree on 84/180 requests and split
   those wins 38–38 (McNemar p = 1.00). GPT-5 nano burned 252,246 output tokens (~1,400 per routing
   decision); Laya produced none. A router sits on the critical path of every request, so this is the
   whole argument for putting a non-generative model there.
2. **Laya is a two-tier router wearing three tiers.** 97% of `small` right, but only **7 of 56 `medium`**
   (recall 0.13). nano fails in mirror image: 41 of 61 `powerful` requests sent to `medium`. Design
   consequence: **do not build a 3-tier router on the base checkpoint** — use two tiers, or make the middle
   an escalation decision rather than a classification.
3. **The tier wording is most of the accuracy.** Rewording the tier descriptions, same model and data,
   swings accuracy 21 points (0.428 → 0.639); with concrete examples its macro F1 passes GPT-5 nano. And
   the gold labels themselves needed auditing: blind labelling puts GPT-5 at 0.817 tier agreement, so the
   ceiling is ~0.82 and a 4-point gap is noise.

Also relevant here: the Laya in that study is the **English 421M checkpoint at 512 ctx on Apple MPS,
FP32**. The same engine on this machine via ggmlc (Q8_0, multilingual) measured **16 ms p50 / 434 q/s**
(`laya.exe bench`), i.e. ~11× faster than the study's 184 ms. The router tax in a step that costs
0.2–2 s is then ~1%.

## 1. Five places a "computer-use API" can be set up

Not mutually exclusive; they answer different questions. Ordered by fit with an MCP-first, local-3090,
real-desktop setup.

### A. Local control plane — one HTTP service, Laya routes, the local driver executes
```
POST /v1/step   {task, observation:{ax_text?, screenshot_ref?, diff?}, history}
  -> Laya: tier, needs_tools, is_sensitive, (typed perception answers)      ~16 ms
  -> executor for that tier                                                 0.2-2 s
  -> {action, tool_calls, evidence, verdict}
GET  /route /questions /health  (same shape as laya_router's service)
```
- Executes through the **local driver already present** (`cua-driver`: background-first input that does
  not steal the user's cursor/focus, SOM element indices, `mode='ax'` text capture, approval gates).
- `cua-driver` is a Hermes tool, not an HTTP service, so this option needs one of: a thin HTTP shim
  around the same primitives, or the MCP route in §C with an HTTP front for headless callers.
- **Fit: highest.** It is the only shape that can actually drive the apps this machine cares about
  (ComfyUI, Blender, local browser) without a sandbox, and the router overhead is ~1% of a step.

### B. Anthropic-compatible endpoint — interop-first
Implement a `messages`-shaped endpoint carrying the versioned `computer_*` tool spec, backed by your own
executor, so existing harnesses (Claude Desktop, LangGraph/vnc-use, E2B-style loops, OpenHands) point at
it unchanged. Reference loop: Anthropic's `computer-use-demo` container (VNC 5900, noVNC 6080, Streamlit
8501, combined 8080), trained at **XGA 1024×768** — coordinates are inferred from the *downscaled*
screenshot, so higher resolutions must be scaled on the way in.
- Laya's seat here is the **middleware**: step router, and the injection classifier slot.
- **Cost:** the protocol requires a vision model as the brain. Laya can never be it. Use this when
  third-party clients must connect, not as the primary path.

### C. MCP-first — the native shape of this stack
Both harnesses already load MCP servers (`laya` is registered in Hermes and OpenClaw). Add a sibling
computer-use server exposing: the primitives (capture/ax/click/type/shell), `route_step`, `verify_step`,
and `screen_ask` (typed questions about screen text). The industry has converged on exactly this:
`open-computer-use` and `vnc-use` both ship as MCP servers over Streamable HTTP.
- **Pros:** no new protocol, agent-mediated, composable with the existing six Laya tools.
- **Limit:** MCP tools are called *by a model*. For unattended/cron automation you still want §A's HTTP.

### D. Sandboxed / remote API — isolation-first
E2B Desktop (open source, Xfce, `.screenshot()/.mouse_move()/.left_click()/.write()/.run_command()`),
Scrapybara (hosted, free tier ~10 h/mo), or the Anthropic reference container. The security guidance is
unanimous and worth taking literally: never on a primary machine, no real credentials, domain
allowlisting, human-in-the-loop on sensitive actions.
- **Combined with A, this gives the router a second axis:** not only *which tier* but *where* — local
  driver for your own apps, sandbox for untrusted web surfaces. That is a natural extension of the
  article's idea, and it is the piece that makes unattended runs safe.

### E. Router in front of *model* calls — the article's literal setup
A local OpenAI-compatible proxy (this study, Switchyard, RouteLLM, vLLM Semantic Router) that classifies
each request and forwards it. In computer use this decides **which model answers a step**, not what to
click. The escalation variant maps 1:1 onto a CUA loop: cheap text step first, a judge decides whether to
escalate to the vision model.
- Evidence for the economics: 40–85% cost reduction at ~95% quality is the reported band for routing, and
  one agent benchmark cut cost 74% by routing only 7% of calls to a frontier model.
- **Caveat:** the tier descriptions and labels must be measured, not assumed (§0.3).

## 2. Recommended shape

**A + C as the product, D as the isolation variant, E inside A, B only if third parties must connect.**

```
caller (Hermes agent | OpenClaw cron | HTTP client)
   |
   +-- MCP tools  (interactive, agent-mediated)          <- C
   +-- HTTP /v1/step, /route, /verify  (headless)        <- A
            |
            +-- Laya: tier + is_sensitive + needs_tools + typed perception   16 ms
            |        (quarantined: typed outputs only, no free text)
            |
            +-- executor by tier:
                   T0 no-model   replay / a11y-diff already answers
                   T1 local text AX-text decisions, no screenshots
                   T2 vision     screenshots + grounding (small VLM -> frontier)
                   T3 sandbox    same as T2 but in an isolated desktop
            |
            +-- verify: a11y diff -> Laya typed questions ("did it change as intended?")
```

Two tiers, not three: from §0.2, the base checkpoint cannot hold a middle tier. Make the middle an
**escalation** outcome ("T1 was not confident or the diff did not confirm → go to T2") rather than a
label Laya must predict.

## 3. What Laya should and should not do here (measured on this machine)

| role | verdict | evidence |
|---|---|---|
| Step/tier router, sensitivity + needs-tools flags | **yes** | article: 0.867 needs_tools, ECE 0.093; 16 ms local |
| Quarantined typed perception (bool/enum about screen text) | **yes** | no text generation by construction — the UCM/Dual-LLM requirement |
| Verification of a step from an a11y diff | **built, advisory, measured low** | [`verify-step.md`](verify-step.md): 15 real pairs, 103 typed questions; 0.602 vs a 0.569 majority-class baseline (and 0.733 on the error question, where the constant answer wins); the diff is cheap, the question set is not (~4,600 tokens for 7 questions) |
| Gate on raw screen text | **not yet** | measured false positives to 1.000 on ordinary UI text, false negatives to 0.001 on a real payload, flips on rewording |
| Grounding / click coordinates | **no** | no vision encoder, no spatial output |
| Holding a full a11y tree | **no** | one window = 42,878 chars ≈ 12k tokens vs a ~2,300-char usable window |

## 4. Security layer for the API

- **Quarantine, don't judge.** Route untrusted screen content through typed questions (bool/enum) so
  nothing can return as an instruction. This is architectural and does not depend on calibration.
- **Screen the untrusted regions, structurally.** Chrome/controls are trusted structure; page/document
  text is the untrusted channel. (Raw AX dumps also blow the context budget.)
- **Gate actions before execution** — irreversible/sensitive action strings need a typed decision plus
  human confirmation; this is the standing CUA recommendation and is screen-independent.
- **State the boundary:** image-embedded injections never appear in a11y/OCR text. A text-side gate can
  never be the whole defence — a visual check is still required.

## 5. Eval plan (copied from the study's structure)

Borrow the parts that made it credible: one **shared question schema** both backends answer
(`questions.py`), swappable backends behind one endpoint, **labels in a committed data file**, metrics
with confidence intervals, and an **ablation on the wording**. Specifically:

1. Label real steps from your own loops (tier, needs_tools, is_sensitive) — **blind**, then measure label
   agreement with an independent model (the study's ceiling was 0.82).
2. Measure zero-shot Laya against a frontier model on those labels; report paired discordance, not just
   accuracy.
3. Only then fine-tune for the tiers that failed (the middle tier, per §0.2), re-measure, and flip any
   enforcement from advisory to enforce.

## 6. Build order

1. 2-tier router over existing step logs — no screen integration, pure classification. Cheap, honest,
   and it settles whether Laya earns a seat on the critical path.
2. HTTP `/route` + MCP `route_step` so both harnesses can call it (§A/§C).
3. `verify_step` on a11y diffs (§3, row 3) — the highest-value typed use, needs no new model.
4. Sandbox executor (§D) for untrusted surfaces; keep the local driver for your own apps.
5. Fine-tune + eval for anything that gates or blocks.