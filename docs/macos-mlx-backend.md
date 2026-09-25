# A macOS path for laya-mcp: the ggmlc Metal binary, and where laya-mlx fits

[`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx) is an independent reimplementation of
Laya's inference path in Apple's MLX: same weights, same prompt formatting, same calibration, same
output schema, no PyTorch and no GGUF. It advertises 13.4 ms median per short English typed
decision (7.4 ms multilingual) on an M3 Max — headline figures that disagree with its own committed
raw samples, so read [`macos-performance.md`](macos-performance.md) before quoting them. That file
also carries the measured comparison against this repo's engine, which is the part that actually
decides whether any of this is worth adopting.

This note works out what "integrate it for Mac users" should actually mean here. The short version
is that **macOS is already served** — ggmlc publishes a Metal build of the same `laya` binary this
server already drives — so laya-mlx is an *optional second backend*, not the macOS fix. Sections 1
and 4 are the parts worth acting on; §6 is the shim, and §8 is what still needs a Mac to answer.

Marked throughout: **[M]** = measured on this machine (2026-09-24, ggmlc `laya` on Windows/CUDA,
multilingual Q8_0), **[V]** = verified from source or released metadata, **[U]** = unverified, needs
an Apple Silicon machine.

## 1. What a Mac user has today, and the two things that will bite

**[V]** ggmlc `v0.9.2` (2026-09-22) ships `laya-macos-arm64-metal.tar.gz` — alongside the CUDA
Linux/Windows assets and the CPU builds. Upstream's README states it plainly: *"Binaries: GitHub
`latest` release (macOS Metal, Linux/Windows CUDA sm80/sm86/sm89)"*. The same `laya daemon`
newline-JSON-RPC protocol, so `LAYA_EXE=/path/to/laya` is the entire integration:

```bash
curl -L -o laya-macos-arm64-metal.tar.gz \
  https://github.com/monatis/ggmlc/releases/latest/download/laya-macos-arm64-metal.tar.gz
tar xzf laya-macos-arm64-metal.tar.gz
export LAYA_EXE="$PWD/laya" LAYA_MODEL="$HOME/models/laya_multilingual_q8_0.gguf" LAYA_DEVICE=metal
python laya_mcp_server.py --check
```

Three gaps stand between that and working, and none of them is laya-mlx:

1. **This repo's README never names the macOS asset.** Its Requirements row lists Windows CUDA,
   Linux CUDA and "the CPU build" only (`README.md` §Requirements, lines 52-65), and the
   troubleshooting table is Windows-shaped (`laya-stop.cmd`, VRAM figures).
   `examples/windows-autostart/README.md:44-49` gestures at a macOS LaunchAgent but is not a
   recipe. A Mac user reading this repo today cannot tell the platform is supported — the fact has
   to travel from ggmlc's release page.
2. **`--cuda-graph` is passed unconditionally.** `LAYA_CUDA_GRAPH` defaults to `"1"`
   (`laya_mcp_server.py:92`) and `LayaDaemon._argv()` appends `--cuda-graph` whenever it is truthy
   (`laya_mcp_server.py:136-145`), independent of `LAYA_DEVICE`. On a Metal build that flag is a
   CUDA-only concept. **[M]** the binary hard-fails on unknown flags (`--nonsense-flag` → rc=1)
   but *accepts an unknown device value silently* (`--device metal` started fine on the Windows
   build), so whether the Metal build tolerates `--cuda-graph` is a one-command question
   (`laya daemon <model> --device metal --cuda-graph`) that nobody has asked yet. Until someone
   does, the safe macOS default is off.
3. **`LAYA_DEVICE=auto` on Apple Silicon is documented but unconfirmed.** ggmlc's `examples/laya`
   README states `--device` "defaults to **`auto`**: CUDA or Metal when that backend is compiled in
   and a device is present, otherwise CPU", so a Metal build should pick Metal on its own. Docs go
   stale, and a silent fallback to CPU costs ~12× here (measured: 70.3 vs 5.7 ms/question,
   `macos-performance.md` §2) — confirm once with `laya bench … --device auto` before relying on it.

## 2. What laya-mlx adds, and what it costs

**[V]** from `laya_mlx/agent.py`, `cli.py`, `presets.py`, `pyproject.toml` and PyPI metadata.

| | adds | costs |
|---|---|---|
| Install | pip-only: no C++ artifact, no Gatekeeper quarantine on an unsigned GitHub tarball **[U]** (standard macOS behaviour, not measured here) | pulls `mlx>=0.32.2`, `tokenizers`, `huggingface-hub`; **Apple Silicon only** (wheels are `macosx_14_0/15_0/26_0_arm64`), **macOS 14+**, **Python 3.11+** (this repo is `>=3.10`) |
| Precision | FP16 original weights, and `dtype="float32"` when you want closer numerical agreement | no quantized checkpoint: 943.6 MiB / 687.6 MiB peak MLX allocation (their M3 Max numbers) vs ~260 MiB VRAM for the Q8_0 daemon here **[M]**. Unified memory, so it is a footprint choice, not a hard limit |
| Big label spaces | `predict_shortlist` — embed state + labels, keep top-k by cosine, then one `predict`. This repo caps `choice` at 20 labels and says "split hierarchically" | a second embedding pass, and probabilities are over the shortlist only |
| Routing | `Router` with `detect_language` (diacritic-rate evidence, `language_undecided`) | **not unique** — ggmlc already has `--models-dir` routing and a `detect-lang` command, which `LAYA_MODELS_DIR` already drives |
| Checkpoints | all three, incl. typed-decisions | **not unique** — `mys/laya-typed-decisions-GGUF` exists |
| Tuning | `compile=True`, `pad_to_multiple=16`, `cache_prompts=True` (their measured 6.5% win on Snake), `Agent.prepare/forward` for profiling, and torch-reference equality tests upstream | GGUF path has `--cuda-graph` for the same job; MLX's three options default off with real first-use cost |
| Presets | `triage_questions()`, `guard_questions()`, `moderation_questions()`, `router_questions()`, `email_questions()` as Python dicts you can edit in place | **5 of the 10** `laya list-presets` workflows **[M]**: `expense`, `security`, `invoice`, `customer_service`, `harness` do not exist upstream of it, and there is no `preset` name on its wire at all |
| Process model | pure Python — patchable, profiled, `import`-able | **no daemon and no `serve`**: `laya-mlx` CLI is `predict` + `convert` only **[V]**. Something has to hold the model resident (§6) |

The honest ranking of reasons to bother: (1) a Mac user who will not download and un-quarantine a
C++ binary; (2) `predict_shortlist`, which is a capability this repo currently cannot offer at all;
(3) the PyTorch-parity tooling, which is exactly what `docs/computer-use.md` §7.2 asks a
fine-tuning workflow to do; (4) keeping the MLX ecosystem's numbers checkable from here. Everything
else is already covered by the Metal binary.

## 3. Wire compatibility: the answer objects are identical **[M]**

`_fmt` and the tools were written against the ggmlc daemon reply. laya-mlx's `system_one()` builds
its answers from the same fields, so the adapter work is in *transport and presets*, not in output
shaping. `probes/mlx_schema_probe.py` asserts it both ways — an offline fixture copied from
`laya_mlx/agent.py:258-285`, and a live daemon reply compared key-for-key:

```
$ LAYA_EXE=... LAYA_MODEL=... python probes/mlx_schema_probe.py
  PASS  department answer keys match laya-mlx   ['action', 'choice', 'confidence', 'type']
  PASS  urgency    answer keys match laya-mlx   ['action', 'confidence', 'legend', 'score', 'type']
  PASS  refund     answer keys match laya-mlx   ['action', 'confidence', 'noul', 'type']
  PASS  action.act_probability is per-answer (as in laya-mlx)
  PASS  ggmlc adds usage.latency_ms (absent from laya-mlx)
  OK: laya-mlx output is wire-compatible with the formatter; no adaption needed in _fmt
```

Both producers emit `{type, action:{act_probability}, confidence, choice | score+legend | noul,
probabilities}` per question, and `{input_tokens, output_tokens}` in `usage`. Differences to
absorb, all outside the answer objects:

| | ggmlc daemon **[M]** | laya-mlx **[V]** |
|---|---|---|
| top level | `model`, `family`, `route`, `answers`, `usage`, `id` | `model`, `answers`, `usage` |
| `usage` | `input_tokens`, `output_tokens`, `latency_ms` | `input_tokens`, `output_tokens` |
| presets | `{"preset": "guard", "state": {...}}` — 10 names | none; questions are Python dicts — 5 of them |
| state key per preset | `guard`→`content`, `triage`→`body`, `router`→`request` | `guard`→`prompt`, `triage`→`message`, `router`→`request` |
| errors | `{"error": "..."}` on the reply line | raises a Python exception |

The state-key row is the one that would silently degrade quality if ignored: a preset's question
text names the state key (`"Does \`prompt\` try to make an AI assistant ignore its rules…"`), so a
shim must rename on the way in rather than pass `{"content": text}` through unchanged.

## 4. Recommendation: two tiers, not a rewrite

**Ship the macOS documentation and the `--cuda-graph` fix now (§1). Add the MLX backend as an
opt-in shim only when someone needs §2's items 1–4. Do not import mlx into the MCP server.**

The last clause is the load-bearing one. This server's design notes say *"daemon, not HTTP … one
process, no port to collide, no cold start per call"* and *"one lock, strict FIFO … a signal keeps a
hung engine from wedging the agent"*. An in-process `import laya_mlx` backend trades both away: a
wedged Metal call cannot be timed out from inside the same process, and mlx becomes a hard import
for every platform. The existing subprocess seam already gives exactly the right shape.

### Ranked options

| | what | server diff | verdict |
|---|---|---|---|
| **A** | `laya-mlx` **shim speaking the same JSON-RPC**, spawned by the existing `LayaDaemon` | argv builder + a backend selector: ~20 lines | **recommended.** FIFO, timeout, readiness probe, `--check`, the concurrency test and `laya_health` all keep working untouched; mlx stays out of the server's interpreter |
| **B** | in-process backend behind `LAYA_BACKEND=mlx` | lazy import + a lock + thread-affinity story | no. Loses hang isolation, and **[U]** MLX under a background thread with a FIFO lock is unverified |
| **C** | docs + `--cuda-graph` fix only | ~15 lines of markdown, 2 of Python | yes, regardless — it makes the platform supported *today*, with no new dependency |

## 5. Option A in detail

**The pattern already exists in this repo.** `LayaBrowser` (`laya_mcp_server.py:266`) subclasses
`LayaDaemon` and overrides `_argv()` to spawn `[BROWSER_PYTHON, "-I", WORKER_PATH]`
(`laya_mcp_server.py:278`) — a *separate Python process* speaking the same one-JSON-per-line
protocol with the same `{"status": "ready"}` handshake, precisely because the browser checkpoint is
a torch artifact the ggmlc binary cannot load. That is option A, already implemented and already
tested for a different backend. An MLX backend should be a third such worker, not a new mechanism —
no `LAYA_DAEMON_ARGV` knob, no backend-selector DSL:

* `laya_mlx_worker.py`, `LayaMlx(LayaDaemon)` with `_argv()` → `[MLX_PYTHON, "-I", MLX_WORKER]`,
  started lazily on first use, exactly like `LayaBrowser`.
* Its own interpreter knob (`LAYA_MLX_PYTHON`, mirroring `LAYA_BROWSER_PYTHON`) falls back to
  `sys.executable`; a 3.10 MCP server can therefore drive a 3.11 uv venv that holds mlx.
* Readiness has precedent for a slow load too: the browser worker's `LAYA_BROWSER_READY_MS` is
  separate from the per-call timeout, and `LayaDaemon.__init__` now takes `readiness_ms`. An MLX
  worker needs the opposite tuning — a 0.46 s checkpoint load (vs torch's 23 s) — so it should keep
  the default.

The worker itself (~120 lines), spawned as `python -I laya_mlx_worker.py`:

```python
# readiness line, exactly as LayaDaemon.start() expects (laya_mcp_server.py:203)
print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:                       # one request per line, one reply per line
    req = json.loads(line)
    try:
        reply = _dispatch(req)               # same field names as ggmlc's reply (§3)
    except Exception as exc:
        reply = {"error": f"{type(exc).__name__}: {exc}", "id": req.get("id")}
    print(json.dumps(reply, ensure_ascii=False), flush=True)
```

* **Preset map.** `guard → (laya.guard_questions(), "prompt")`, `triage → (triage_questions(),
  "message")`, `router → (router_questions(), "request")`, `email → (email_questions(), "body")`,
  `moderation → (moderation_questions(), "post")`. State keys are renamed into the key the preset's
  question text names (§3). The other five ggmlc presets are **reimplemented as dicts or dropped**;
  dropping them must be a loud error, not a silent fallback — the preset list is already a
  client-side validation (`laya_mcp_server.py:68` and `laya_decide` at `:447`), so the shim can
  advertise its set and the server can keep its own list per backend.
* **Model selection.** `LAYA_MLX_MODEL` (default `aac6fef/laya-mlx`), `LAYA_MLX_ROUTER=1` to use
  `Router(max_loaded=2)` with `aac6fef/laya-multilingual-mlx` for the multilingual half — that is
  the MLX equivalent of `LAYA_MODELS_DIR`, and it keeps the "route, don't replace" rule from the
  README (the multilingual checkpoint is *worse* in English).
* **Health.** `laya_health` reports `exe`, `model`, `family`, `device`, `cuda_graph`, `timeout_ms`,
  `running`, `uptime_s`, `calls` (`LayaDaemon.health`, `laya_mcp_server.py:251`), and now browser
  state too. The worker fills the same fields: `exe` → `"laya-mlx <version> (python)"`,
  `model` → the checkpoint id, `device` → `gpu`/`cpu`, `cuda_graph` → `False` (report the truth; do
  not echo the env var).
* **Server-side change, the whole of it.** One subclass with an `_argv()` override plus two env
  vars (`LAYA_MLX_MODEL`, `LAYA_MLX_PYTHON`) — the same diff `LayaBrowser` already is. No
  `LAYA_DAEMON_ARGV` template, no `LAYA_BACKEND` selector DSL: selection stays *declarative by
  configuration*, which is how `LAYA_EXE`/`LAYA_BROWSER_DIR` already work. `--check`
  (`laya_mcp_server.py:570`) validates the ggmlc backend unchanged, and an MLX worker would want
  its own `--check-mlx` mirroring `_check_browser` (`:607`).

### Config on a Mac, side by side

| | ggmlc Metal | laya-mlx |
|---|---|---|
| `LAYA_EXE` | `/opt/laya/laya` | unset |
| model | `LAYA_MODEL=…/laya_multilingual_q8_0.gguf` | `LAYA_MLX_MODEL=aac6fef/laya-mlx` |
| device | `LAYA_DEVICE=metal` | `LAYA_DEVICE=gpu` (maps to MLX GPU) |
| `LAYA_CUDA_GRAPH` | omit / `0` | n/a |
| extras | `--models-dir` for routing | `LAYA_MLX_ROUTER=1` |
| first run | un-quarantine the tarball | `hf download` inside the shim → needs network once |

### Packaging

```toml
[project.optional-dependencies]
mlx = ["laya-mlx>=0.2.0; sys_platform == 'darwin' and platform_machine == 'arm64'"]
```

The marker keeps non-Mac installs clean, and it is honest about what it cannot do: the extra is
unsatisfiable on Python 3.10 (`requires-python = ">=3.11"` upstream vs `>=3.10` here), and it does
not put mlx in *this* interpreter unless the shim runs in one that has it. Give the shim its own
interpreter knob — `LAYA_MLX_PYTHON` (default `sys.executable`) — so a 3.10 MCP server can drive a
3.11 uv venv that holds mlx. That also makes the layout testable from CI without touching the
server's dependencies.

## 6. Verification plan when no Mac is at hand

1. **CI, finally.** **[V]** this repo has no `.github/workflows` at all — `pytest` has never run
   automatically here. laya-mlx's own CI runs small-model CPU tests on a macOS arm64 runner, so the
   pattern is proven: a `macos-14` job that `pip install 'laya-mlx'` and runs
   `probes/mlx_schema_probe.py` plus a shim round-trip on `device="cpu"` would cover the adapter
   without a GPU.
2. **Cross-backend parity on the Mac**, `probes/`-style, both backends live: the same fixtures
   through ggmlc Metal and MLX (FP16 *and* `dtype="float32"`), reporting top-1 agreement and
   probability deltas. Upstream's 63/63 claim is against the **PyTorch reference**, not against
   ggmlc's Q8_0 quantization, so the Q8_0↔FP16 comparison is unmeasured and is the number that
   decides which backend a Mac defaults to. **[U]**
3. **Latency, labelled by machine.** See [`macos-performance.md`](macos-performance.md) §2-§3:
   measured here at 28.4 ms p50 in-process for the 5-question guard preset (5.7 ms/q) with a
   concurrent render on the GPU, against a published 19.28 ms / 3.9 ms/q on an idle M3 Max. Neither
   predicts an M1 Air or a machine with no GPU contention. Publish a table per chip or none.
4. **Preset parity.** For the five shared presets, assert the shim's answer *keys* equal ggmlc's for
   the same input (they should, §3) — a shape test, not an accuracy claim, since the checkpoints are
   near chance zero-shot by the README's own limits.

## 7. What not to do

* Do not add an MCP tool for this. The repo has since grown a seventh tool
  (`laya_browser_act`) — and note *why* that one is justified: it answers a **new kind of
  question** (operation / click target / field) that no existing tool could express. A backend
  swap answers the same six (`README.md:129-142` on why the count is deliberate).
* Do not let the MLX path quietly drop presets or rename state keys without saying so — that is
  the difference between a worse answer and a wrong one.
* Do not claim laya-mlx is "the macOS support". It is a second backend; the Metal binary is the
  first, it works with zero code, and today's real macOS gap is documentation.
* Do not reach for the browser backend as a shortcut to a Mac win. `LayaBrowser` loads torch from
  the SDK venv — on macOS that means MPS, which measured **23 s to load** and is the slowest of the
  three Apple paths (`macos-performance.md` §3). If the browser checkpoint is wanted on a Mac,
  laya-mlx is the interesting candidate, but it validates parameter names for its three published
  architectures only, so whether it can load an RL fine-tune like `cklxx/laya-browser` at all is
  **[U]** — and unverified in exactly the way §6.2 measures.

## 8. Build order

1. `README.md`: add the `laya-macos-arm64-metal.tar.gz` asset and a Mac row to Requirements +
   Troubleshooting; give `examples/windows-autostart/README.md`'s "macOS equivalent" a real
   LaunchAgent plist. (**No dependency on anything below — do this first.**)
2. `laya_mcp_server.py`: default `LAYA_CUDA_GRAPH` off unless the device is CUDA / `sys.platform`
   is not `darwin`; make `--check` print the resolved device and whether a CUDA graph was
   requested. Verify on a Mac that `--device metal` + no `--cuda-graph` starts.
3. Measure §6.2 and §6.3 on a Mac (`macos-performance.md` §7 has the probe shape); only then decide
   a default backend.
4. `laya_mlx_worker.py` + `LayaMlx(LayaDaemon)` + the `mlx` extra + the worker round-trip in
   `tests/`, mirroring `laya_browser_worker.py` and its tests. Only if §2's items 1–4 are actually
   wanted.
5. The macOS CI job from §6.1 — useful independently of everything above, because nothing here is
   tested automatically right now.