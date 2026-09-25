# Game playing as a step policy: what the laya-mlx Snake demo gives laya-mcp

The [Snake demo](https://deepwiki.com/mizorewww/laya-mlx/3-snake-demo) in
[`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx) is the first complete *loop* built on
a Laya checkpoint: an environment, a policy, a safety shield, recording, and a benchmark — with
published numbers. This repo has the decision engine and the agent-facing tools, but no loop of its
own. This note works out what to take from the demo and what the integration should look like as a
general game-play / step API — the same surface computer use will need later.

Evidence script: [`probes/snake_loop_probe.py`](../probes/snake_loop_probe.py) — the port is
runnable, so every number in §3 can be re-measured on your own engine.

Marking: **[M]** measured on this machine (RTX 3090, multilingual Q8_0 via ggmlc, 2026-09-25),
**[V]** verified from upstream source or docs, **[U]** unverified.

**The short version:** take the *loop mechanics* (planner features → one batched typed call →
admissible-only execution → JSONL record → sweep/soak benchmark), not the terminal UI and not the
MLX dependency. Ship them as `laya_policy/` with Snake as the reference environment: a Python
package for loops, an HTTP `/step` for everything else, and **no new MCP tool** until a measured
workflow needs one.

## 1. What the demo is, part by part

| part | upstream | what it does |
|---|---|---|
| engine | `laya_mlx/snake/game.py` | deterministic rules + a Hamiltonian-cycle planner: every direction is classified legal / safe / dead-end, with cycle progress and current food reachability **[V]** |
| policy | `laya_mlx/snake/policy.py` | one compact state string + **one batched call of three questions**: a `choice` over the four directions whose *criteria are the planner's own descriptions* ("Safe. Best route to food." / "Unsafe. Traps the snake." / …), plus two `noul`s — "Is a safe route available?" and "Is food reachable through empty cells?". The model never sees the board; it ranks described options **[V]** |
| shield | same | executes the highest-probability **admissible** direction; an inadmissible raw top-1 is corrected, the raw probabilities are kept, and the intervention is counted. `--unassisted` disables it. Guarded play from a valid board cannot die and makes bounded food progress (cycle invariant + tests) **[V]** |
| record | `run.jsonl` | per-frame boards, **original model probabilities**, executed actions, timings, provenance (`laya-snake-v1`); replay is deterministic and the published runs are replayed frame-for-frame in the test suite **[V]** |
| bench | `benchmark.py` | rate sweep → 4-seed soak; a rate passes only if ≥99% of active ticks fit the budget on **every** seed, and failures are kept in the shipped results **[V]** |

Published headline numbers (M3 Max, FP16, multilingual) **[V]**: **63.61 moves/s uncapped** over
2,400 moves, zero deaths; model-inference p50 10.08 ms/move; highest passing paced budget 20 FPS
(18.7–18.9 moves/s actual, 99.75% of ticks within budget); an opt-in optimized path +6.5%; the
30-second showcase ran 1,144 moves at 11.44/s, score 40.

Keep the line upstream draws: this is **feature-assisted**. "Neither mode establishes that the
checkpoint can infer Snake strategy from an unprocessed board" — the planner does the reasoning,
the model ranks and confirms **[V]**. That is the same measured-advisory discipline this repo runs
on, and it should survive the port intact.

## 2. Why this matters here

- **A loop testbed with ground truth.** Every aux question has a computable truth (safe route?
  food reachable? move outcome), and the game ends in score/len/death. The repo's next measured
  steps — fine-tuning (#3 step 5), question-set shrinkage and calibration (#10) — all need exactly
  that, and games generate it for free.
- **The computer-use ladder needs the same mechanics at lower stakes** (decide → execute → record →
  verify). Snake is the cheapest place to debug those mechanics because failures cost nothing and
  are replayable.
- **Cross-engine.** Upstream is Apple/MLX-only. Through the daemon, the same loop runs on
  Windows/Linux CUDA and macOS Metal, and gives a same-workload comparison against the published
  MLX numbers.

## 3. Measured on this machine: the loop runs on ggmlc

Port: [`probes/snake_loop_probe.py`](../probes/snake_loop_probe.py) — the demo's game rules and
compact wording verbatim (attributed), driven through this repo's daemon seam, headless.

Quiet GPU (1% util, 5.2/24.5 GB used):

| mode | seed | moves | moves/s | call p50 / p99 | engine p50 | score (len) | interventions |
|---|---|---:|---:|---|---:|---|---:|
| shield | 7 | 300 | **174.93** | 5.3 / 6.0 ms | 5.1 ms | 6 (12) | 0 |
| shield | 101 | 300 | **174.67** | 5.3 / 5.9 ms | 5.1 ms | 6 (12) | 0 |
| raw | 7 | 300 | **172.09** | 5.4 / 6.1 ms | 5.2 ms | 6 (12) | — (off) |
| raw | 101 | 300 | **174.40** | 5.3 / 6.2 ms | 5.1 ms | 6 (12) | — (off) |
| shield | 7 | 1000 | **171.21** | 5.5 / 6.3 ms | 5.3 ms | 22 (28) | 0 |

All runs: zero deaths, zero invariant violations. An earlier set of runs while the GPU was
otherwise saturated (100% util, 23.7/24.5 GB used — a ComfyUI-class job) measured the same loop at
**60.0–62.3 moves/s** with call p50 15.5–16.7 ms: the spread is the machine, not the loop.

Notes, all **[M]**:

- **Same order as upstream.** 171–175 moves/s on this engine vs their published 63.61 uncapped
  (which included terminal rendering; ours includes daemon I/O and excludes rendering). Per-frame
  model time p50 5.1 ms for the whole 3-question batch vs their 10.08 ms/move. Different stacks —
  treat as the same order of magnitude, not a like-for-like win.
- **The daemon round trip is ~0.2 ms** here (call p50 5.3 vs engine 5.1); the engine dominates.
  152.8 input tokens/frame → ~51 tokens/question.
- **Raw vs shield: identical trajectories** on the shared seeds — the model's raw top-1 was
  admissible at every frame (0 interventions in ~3,400 measured moves; upstream sees ~4 in 8,160).
  Rarity is expected; exercising the shield needs a seed sweep or a longer soak. `--unassisted`
  already drives the same loop without the shield.
- **Aux ground truth is degenerate on short runs** — food reachable and safe-route were "yes" on
  every frame, so "answer accuracy 1.000" is vacuous and is deliberately not quoted as a result.
  It gets teeth on mid-game boards (long runs) or off-cycle play; the probe records it either way.
- **Contention reality check:** a fresh daemon's first calls under severe GPU load exceeded the
  30 s default call budget; the probe's `--timeout-ms 120000` absorbed it, after which steady-state
  p99 stayed ≤17 ms. Loop callers should budget a generous first-call timeout.
- Score ran lower than upstream's soaks (22/1000 vs 16–24/600). Candidates: Q8_0 vs FP16 choice
  shifts (the Q8_0↔FP16 comparison is exactly what issue #6 §6.2 leaves unmeasured), or seed luck.
- Upstream's opt-in MLX compilation/prefix-cache path (+6.5%) has no equivalent here; the ggmlc
  knob is the CUDA graph, already on in these runs.

## 4. Options considered

| | option | verdict |
|---|---|---|
| A | **Port the loop into a new `laya_policy/` package** (env contract + committed spec + runner + record + eval + CLI; Snake as reference env; HTTP `/step` as its second surface) | **recommended** — it is the piece #3's build order already implies, at the cheapest possible risk |
| B | Docs-only write-up, no code | insufficient — the repo's own precedent pairs docs with probes, and the loop mechanics are cheap to port and immediately measurable |
| C | Import the demo as-is (terminal UI, rich, MLX-only) under `games/snake/` | no — duplicates upstream, Apple-only; the presentation layer is not the product here |
| D | Ship the step call as a ninth MCP tool (`game_step`) | not now — loops are programmatic, not agent-mediated; the eight-tool discipline stands; `laya_decide` already covers agent-side single asks. Revisit only with a concrete workflow that needs it |
| E | Go straight to the full computer-use step API (screen env + executor) | no — #3 puts the sandbox executor and fine-tune after measurement; Snake first is the honest testbed |
| F | Contribute the port upstream instead | no — upstream is deliberately MLX-specific; the engine-agnostic policy layer belongs where the daemon seam already lives |

## 5. Recommended shape

### 5.1 Package layout (mirrors `laya_router/`)

```
laya_policy/
  __init__.py
  spec.py            # load / validate / digest a policy spec (data, not code)
  specs/snake.json   # the committed spec: state template, questions, variants
  env.py             # the environment contract (see 5.2)
  envs/snake.py      # port of game.py (attribution) + adapter to the contract
  policy.py          # decide(): spec + env -> questions -> backend -> shield decision (one frame)
  record.py          # laya-policy-v1 JSONL writer/loader + deterministic replay check
  run.py             # headless loop CLI (--env snake --moves N [--fps F] [--raw] [--record])
  eval.py            # sweep/soak/budget benchmark + aux accuracy/calibration
  service.py         # loop-facing HTTP: /health /spec /step (stdlib, mirrors laya_router.service)
  data/results/      # committed run outputs (snake_ggmlc.json, …)
tests/test_policy.py # cycle invariants, shield semantics on a stub backend, record round-trip
```

Reuses, not new mechanisms: the `LayaDaemon` transport seam (as the probes already do), the
`laya_router.questions` spec/digest pattern, `laya_router.backends.LayaBackend` for backend
selection, and `laya_router.service`'s stdlib HTTP + error discipline (502 on unanswered). No
rich/UI dependency — presentation stays upstream's job.

### 5.2 The environment contract (five methods)

```python
class Env(Protocol):
    def observe(self) -> dict: ...            # features for the state template (bools/ints only)
    def options(self) -> dict[str, str]: ...  # action -> description, from the spec's criteria map
    def admissible(self) -> list[str]: ...    # the shield's allow-list, from the planner
    def step(self, action: str) -> dict: ...  # {"done": bool, "score": int, ...}
    def ground_truth(self) -> dict: ...       # truth for the aux questions this frame (optional)
```

The *wording* lives in the spec, not the code: the router study measured 21 accuracy points from
re-wording alone, so the spec is a data file with a digest that goes into every trace. Ported
strings stay identical to upstream's compact prompt until an ablation says otherwise — the wording
is part of the measurement.

### 5.3 One frame = one call, ≤4 questions

Same shape as the demo: one `choice` (the options) + up to three `noul`s (aux checks). The encoder
pays `state × questions` (#10), so keep game questions short and batched; the token count per call
is recorded (`input_tokens`) so cost drift is visible.

### 5.4 Shield semantics — and where enforcement stops

- **Games: enforced.** Execute argmax over the admissible set; keep raw probabilities; set
  `intervened` when corrected. The admittance set is the environment's own safety property; this is
  what makes guarded play provably alive.
- **Real-machine steps (later): advisory.** The same API returns `advisory: true` and `intervened`
  as *evidence*, never as a block; the driver decides. No game-side enforcement ever leaks into
  computer-use gating — the two risk domains are not the same.

### 5.5 Record format (`laya-policy-v1`)

```jsonl
{"type":"metadata","format":"laya-policy-v1","env":{"name":"snake","seed":7},
 "spec":{"name":"snake","version":"snake-v1","digest":"…"},
 "backend":{"model":"laya_multilingual_q8_0.gguf","device":"cuda"},"policy":{"guarded":true}}
{"type":"frame","tick":42,"probabilities":{"UP":0.07,"DOWN":0.04,"LEFT":0.82,"RIGHT":0.07},
 "proposed":"LEFT","executed":"LEFT","intervened":false,"aux":{"risk":0.03,"food":0.92},
 "aux_truth":{"risk":true,"food":true},"ms":{"wall":5.3,"engine":5.1},"tokens":153}
{"type":"end","summary":{"moves":300,"score":6,"interventions":0,"moves_per_second":174.9}}
```

Deterministic replay is a test, not a demo: same actions + seed must reproduce snapshots, in the
upstream test-suite style.

### 5.6 Eval harness

Port the benchmark's sweep → soak criterion verbatim (≥99% of active ticks within budget, every
seed; failures kept), then extend it with the part upstream does not have: **aux accuracy and
calibration against ground truth** (per-question accuracy vs the game's computation, mean P(true)
when true vs false, ECE). Run outputs are committed like `laya_router/data/results/` — same
config-and-fingerprint discipline.

### 5.7 Surfaces

- **Python + CLI (primary; loops):** `python -m laya_policy.run --env snake --moves 600 --seed 101
  [--fps 12] [--raw] [--record run.jsonl]`; `python -m laya_policy.eval --env snake --rates
  10,20,30,60 --seeds 101,102,103,104`.
- **HTTP (cross-language/headless):** the stateless step oracle — the daemon contract plus the
  shield's two additions:

  ```jsonc
  // POST /step
  { "state": "Safe route: yes. Food reachable through empty cells: yes.",
    "questions": { /* or {"spec": {"name":"snake","variant":"compact"}} */ },
    "admissible": ["LEFT", "UP"],   // optional; shield executes over this set
    "mode": "guard" }               // or "raw"
  // -> {"probabilities": {…}, "proposed":"LEFT", "executed":"LEFT", "intervened":false,
  //     "aux": {…}, "spec_digest":"…", "advisory":false, "latency_ms":5.3, "input_tokens":153}
  ```
- **MCP: none for now.** `laya_decide` already covers agent-mediated single asks. A ninth tool
  waits for a measured workflow — the tool-count argument in `README.md` applies unchanged.
- Run it as its **own process** (own daemon, ~260 MiB): a hot loop should not share the MCP
  server's daemon behind its FIFO lock.

### 5.8 Computer-use continuity

`envs/screen.py` (later): observation = accessibility-region text or a diff; options = candidate
operations/elements; aux = typed perception ("did the dialog appear?"); admissible = an action
allowlist; `advisory: true` throughout, execution stays with `cua-driver`. That is the T1/T2 step
of `computer-use-api.md` §2, and the loop mechanics it needs are exactly what this port benches.
One boundary carries over unchanged: a text channel cannot see image-embedded instructions
(`computer-use.md` §6) — and note the demo's questions run on *planner-authored* text, not raw
screen text, which is why the wording-calibration problems in #10 are cheaper here.

## 6. Build order

1. **Probe (landed with this note):** `probes/snake_loop_probe.py` — §3's evidence, re-runnable.
2. **This document** + the probe row in `probes/README.md` (landed).
3. **`laya_policy/` core:** spec/env/policy/record/run + tests. Acceptance: `pytest -q` green
   without a model (stub backend); `run --env snake --moves 300 --seed 7` reproduces §3's shape.
4. **Eval:** sweep/soak/budget + aux scoring; first committed results file. Acceptance: one
   command regenerates a committed JSON with config + source fingerprint.
5. **Service:** `/health`, `/spec`, `/step` with the router's error discipline. Acceptance: a curl
   example in this doc returns §3-shaped decisions.
6. **Second env (when wanted):** bundle flappy (the recipe already exists in
   `probes/flappy_recipe_probe.py`) to prove the contract is not snake-shaped. Then `envs/screen.py`
   for #3 steps 4–5.
7. **Calibration experiment:** refit one temperature per (question type, option count) on recorded
   frames; report ECE before/after on a held-out split. The first temperature refit *with ground
   truth from a running loop* — the thing #10 asks for, on the cheapest corpus available.

## 7. Open questions for the owner

- Package name: `laya_policy` (recommended) vs `laya_play`? The loop layer is engine- and
  game-agnostic either way.
- Stateless `/step` in v1, or sessions now (for a non-Python client that wants the server to own
  the env)? The stateless shape covers every caller in sight.
- Is publishing the ported numbers as a table — this machine vs upstream's M3 Max — useful, and is
  a like-for-like Mac run worth chasing via #6?
- Should anything of the terminal presentation be revived (a tiny `--print` trace for humans), or
  left entirely upstream?

## 8. Honest limits

- **Feature-assisted, not strategy.** No claim the checkpoint understands Snake; the planner
  describes, the model ranks. Zero-shot bespoke typed questions measure near chance in both
  projects; presets and planner-described options are what work.
- **Uncalibrated probabilities.** The shipped checkpoints are over-confident by construction;
  §5.7's temperature refit is required before any number is used as a gate.
- **The §3 aux ground-truth result is degenerate on short runs** (all-yes). It is recorded, not
  quoted; longer/unshielded runs are the follow-up.
- **Load sensitivity is large:** 60→175 moves/s between machine conditions. Any published figure
  needs the conditions attached.
- **No new safety properties:** the shield is the game's own determinism. Nothing here gates a
  real machine; computer use stays advisory until a measured eval (#3).
- **One daemon per process** (FIFO); budget the VRAM and the first-call timeout.

## References

- DeepWiki: [Snake Demo](https://deepwiki.com/mizorewww/laya-mlx/3-snake-demo) ·
  [Game Engine and UI](https://deepwiki.com/mizorewww/laya-mlx/3.1-snake-game-engine-and-ui) ·
  [Decision Policy](https://deepwiki.com/mizorewww/laya-mlx/3.2-snake-decision-policy) ·
  [Benchmarks](https://deepwiki.com/mizorewww/laya-mlx/3.3-snake-benchmarks-and-results)
- Upstream: `docs/SNAKE_DEMO.md`, `docs/SNAKE_BENCHMARKS.md`, `docs/SNAKE_OPTIMIZATION.md`,
  `laya_mlx/snake/*` (Apache-2.0; the port carries attribution headers).
- This repo: `docs/computer-use.md`, `docs/computer-use-api.md` (#3), `docs/verify-step.md` (#10),
  `docs/macos-mlx-backend.md` (#6), `laya_router/` (spec + digest discipline).