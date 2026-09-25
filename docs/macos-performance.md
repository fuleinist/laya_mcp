# Performance: this repo's engine vs the Apple Silicon runtimes

What the numbers actually say, with the measurement boundary attached to every figure. Two of the
three Mac-side options publish numbers; the one that needs no new code publishes none, and that
absence is the honest headline.

Measured **[M]** 2026-09-24 on this machine: RTX 3090, Windows, ggmlc `laya` v0.9.2,
`laya_multilingual_q8_0.gguf` (345 MB). Published **[P]** figures are taken from the projects'
own checked-in benchmark artifacts — raw JSON where it exists, not the README headline.

## 1. The one comparison that matters and cannot be made yet

| Mac backend | installed by | Apple Silicon latency published? |
|---|---|---|
| ggmlc `laya-macos-arm64-metal` | one tarball, no Python deps | **no — zero numbers anywhere** |
| `laya-mlx` (MLX) | `pip install laya-mlx` | yes, M3 Max, self-reported |
| `laya-coreml` (Core ML / ANE) | `pip install laya-coreml` | yes, M3 Max, self-reported, energy too |

**[P]** ggmlc's README documents the Metal asset and nothing else about it: no Metal row in its
`examples/laya` latency tables (every row there is an RTX 4050 Laptop), no Apple Silicon figures
in `BENCHMARKS.md`, no third-party measurement that a search turns up. So "which is faster on a
Mac" is unanswered, and the answer is one command away for anyone with a Mac:

```bash
laya bench ~/models/laya_multilingual_q8_0.gguf --preset guard --device metal --warmup 5 --runs 7
```

## 2. Current setup, measured [M]

RTX 3090, Q8_0 multilingual. **Caveat that runs through this section: ComfyUI was rendering on the
same GPU throughout — `nvidia-smi` reported 100% utilisation and ~23 GB in use.** Every CUDA figure
below is therefore a pessimistic bound, not a best case.

In-process (`laya bench`, the same boundary laya-mlx reports), `guard` preset, 5 questions, S=83, B=5:

| device | p50 | mean | best | ms/question | questions/s |
|---|---:|---:|---:|---:|---:|
| cuda + graph | **28.4 ms** | 28.5 | 28.2 | 5.7 | 175.4 |
| cpu, 8 threads | 353.6 ms | 351.5 | 338.2 | 70.3 | 14.2 |

Through `laya daemon` — the path the MCP tools actually take, so this adds the JSON-RPC round trip
and result formatting:

| workload | tool-level p50 | p95 | in-engine `latency_ms` | transport + scheduling |
|---|---:|---:|---:|---:|
| 1 question, 63 tokens | **18.30 ms** | 23.46 | 7.07 | **11.2 ms** |
| 3 questions, 189 tokens | 25.64 ms | 32.67 | 25.22 | 0.4 ms |
| 5 questions (`guard`), 340 tokens | 38.66 ms | 45.86 | 39.66 | ~0 |
| 50 questions, 3,290 tokens | 244.48 ms (4.89 ms/q) | 255.95 | 214.13 | 30 ms |

Two things fall out of this table that are worth more than the comparison itself:

1. **At one short question the engine is 7 ms and the tool call is 18 ms.** Two thirds of a
   single-question decision is process round trip and scheduling, not inference. The MCP path's
   floor is set by the daemon pipe, not by the model — which is exactly why the harness-level
   prompt in `README.md` treats these tools as cheap-but-not-free.
2. **The per-question cost falls 3.7× from 1 question to 50** (18.3 → 4.9 ms/q). Any "Laya costs
   X ms" claim without a batch size attached is meaningless, on either backend.

## 3. Apple Silicon, as published [P]

M3 Max, 40-core GPU, 128 GiB, macOS 27.2, MLX 0.32.2, `batch_size=64`, one idle machine. All
figures are end-to-end in-process (prompt prep, tokenization, tensors, synchronized inference,
calibration, formatting; model loading excluded) — i.e. the same boundary as the `laya bench` row
above, *minus* the daemon round trip.

| workload (tokens) | laya-mlx multilingual FP16 | laya-mlx English FP16 | PyTorch MPS FP32 multilingual | this repo, RTX 3090 Q8_0 **[M]** |
|---|---:|---:|---:|---:|
| 1 short question (91-93 tok) | **10.91 ms** in-process (fwd 8.72) | 17.75 ms (fwd 15.82) | 13.60 ms | 7.07 ms in-engine, 18.30 ms as a tool call |
| 5 questions | **19.28 ms** | 44.43 ms | 27.82 ms | 28.4 ms in-process, 38.66 ms as a tool call |
| 10 questions | 32.92 ms | 80.82 ms | 43.87 ms | — |
| 50 questions | **125.49 ms / 402 q/s** | 347.24 ms / 143 q/s | 195.16 ms / 251 q/s | 244.48 ms / 205 q/s |
| ms per question, 5 Q | 3.9 | 8.9 | 5.6 | 5.7 |
| full context, 1 Q | 43.50 ms (1024 tok) | 49.84 ms (512 tok) | 54.39 ms | — |
| model load | **0.46 s** | 0.19 s | 23.16 s | 1.39 s cold first call **[P]** |

Boundaries are **not** identical, and the last column states both of ours so the rows can be read
either way: the published columns are in-process end-to-end (their prompt prep, tokenization,
forward, calibration, formatting — no IPC), our in-process row is `laya bench` for 5/50 questions
and the daemon's own reported `latency_ms` for 1, and our tool-call row adds the JSON-RPC round
trip. Compare like with like: **in-engine at one question the 3090 is ahead (7.07 vs 8.72-10.91 ms)**;
at five and fifty questions the M3 Max is ahead. Most of the gap at 1 question is our own transport,
not either model.

Read against each other, three observations:

- **Neither machine dominates, and the swings are bigger than the differences.** The 3090 is ahead
  in-engine at one question (7.07 vs 8.72-10.91 ms), the M3 Max is ahead at five (3.9 vs 5.7 ms/q)
  and at fifty (402 vs 205 q/s) — and both 3090 rows ran with a render saturating the GPU while the
  M3 Max ran idle. The one consistent signal is that an expensive machine buys single-digit
  milliseconds here, not multiples.
- **A 24 GB discrete GPU's FLOP advantage does not show up here at all.** These are 91-1024 token
  bidirectional encoder passes: launch and scheduling bound, not compute bound. Unified memory plus
  MLX's lazy graph beats a CUDA graph captured for one live shape. Anyone expecting the 3090 to
  dominate should re-measure rather than assume — this is the useful finding of the whole exercise.
- **MLX loads the checkpoint ~50× faster than the PyTorch path** (0.46 s vs 23.16 s), which matters
  if a backend is ever started per-call instead of held resident.

For scale, the reference points outside these two repos: upstream Laya reports **~33 ms per
question on a Tesla T4** and 7.2 ms/question batched; an independent router study measured the
**421M English checkpoint at 184 ms p50 on an M4 via PyTorch MPS** in a 512-token workflow
(`docs/computer-use-api.md` §0); Jev's cloud API is quoted at 236-276 ms. Everything local is
already in the same order of magnitude; the spread between local runtimes is much smaller than the
spread between local and cloud.

## 4. The Core ML / ANE option, and why it does not fit these tools

**[P]** The same author also ships `laya-coreml` (PyPI 0.1.0, `<3.14,>=3.11`), which runs Laya on
the Neural Engine. Its headline is stronger than MLX's, with energy measured off the SMC rather
than modelled:

| metric (one short multilingual decision, M3 Max) | compiled MLX FP16 | Core ML ANE FP16 | Core ML ANE W8 |
|---|---:|---:|---:|
| end-to-end P50 / P95 | 6.94 / 7.39 ms | **4.98 / 5.31 ms** | 4.88 / 5.23 ms |
| mean whole-system power | 61.39 W | 30.75 W | 27.39 W |
| whole-system energy per decision | 0.4288 J | **0.1540 J** | 0.1344 J |
| speed gain | 1× | 1.39× | 1.42× |

**The disqualifier for this repo:** the fast ANE variants have a **96-token total limit covering
instructions, options and state**, and raise a capacity error beyond it. Our `guard` preset alone
consumes 340 tokens of state before the question, and the accessibility-tree text these tools are
pointed at is far larger than that (`docs/computer-use.md` §3). The general-purpose 1024-token Core
ML bundle exists and validated 63/63 on fixtures, but its measured cost is **91.7 ms per
1024-token request** — slower than the plain MLX path, so the ANE's advantage is restricted to a
decision shape our inputs do not have. Fine for a fixed low-cardinality question in a loop; wrong
for a text-screening tool.

## 5. The published MLX numbers do not agree with each other

Worth knowing before quoting any of them. Three values exist for literally the same measurement
(one short multilingual decision, M3 Max, FP16):

| source | value |
|---|---:|
| `laya-mlx` README (and therefore every news write-up) | 7.39 ms P50 / 7.79 ms P95 |
| `laya-mlx` `BENCHMARKS.md` + `benchmarks/results/laya-multilingual-mlx-float16.json` | **10.91 ms P50 / 19.48 ms P95 e2e** (8.72 / 10.00 forward) |
| `laya-coreml`'s comparison table, for "compiled MLX FP16" on the same machine | 6.94 ms P50 / 7.39 ms P95 |

The 7.39 that circulates is laya-mlx's own P50 *and* the sibling repo's P95 *and* neither matches
the raw samples committed in `benchmarks/results/`. **[P]** Every third-party page found in a search
restates the README — one aggregator says outright that the figures "await independent
reproduction". So: use the raw-artifact figures (10.91 / 17.75) when planning, and treat the
headline as a best case measured under a condition that is not documented. The English 421M
checkpoint shows the same gap (README 13.42 vs raw e2e 17.75 ms).

## 6. What to conclude

1. **On a Mac, the existing ggmlc Metal binary is unmeasured but not obviously behind.** Nothing
   published says it is slower than MLX, and it keeps preset parity, `--models-dir` routing and all
   ten workflows. Measure it before adding a runtime.
2. **MLX earns the swap when you want the pip install or the FP16 path**, not for the latency:
   3.9 vs 5.7 ms/q at the 5-question shape is within what a different machine, a different
   precision and a busy GPU can explain.
3. **Neither Mac runtime fixes the actual bottleneck.** At one short question, 11 of our 18 ms is
   the daemon round trip — a Mac backend does not remove that unless it is in-process, and §4 of
   `docs/macos-mlx-backend.md` argues in-process is the wrong trade.
4. **Batch hard on either backend.** 3.7× per-question difference between 1 and 50 questions, which
   is why `laya_classify` takes a list and `README.md` says to call it once rather than loop.

## 7. Settling it on a Mac

`probes/` should grow a cross-backend harness alongside `mlx_schema_probe.py`: same state and
questions, run through `laya daemon --device metal` and through `laya-mlx` in-process, reporting
p50/p95 at 1 / 5 / 50 questions on `device="gpu"` and `device="cpu"`, plus top-1 agreement between
Q8_0 GGUF and FP16 MLX on a fixed fixture. That last number — not latency — decides which backend a
Mac should default to, because the checkpoints are near chance zero-shot (README `Honest limits`)
and a backend that disagrees with the other on the same input is a correctness problem, not a
performance one.