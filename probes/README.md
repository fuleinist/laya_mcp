# Probes

The measurement scripts behind the numbers in `docs/computer-use.md`, `docs/game-playing.md`, and behind the
`laya-game-playing` skill's state-wording rules. They are kept runnable rather than quoted, so any
claim in the docs can be re-measured on your own engine.

| script | what it measures | backs |
|---|---|---|
| `screen_gate_probe.py` | accessibility-tree text through the `guard` preset: clean screen vs planted on-screen injection, and where the window truncates | `docs/computer-use.md` §3 |
| `screen_gate_probe2.py` | the follow-up matrix: prose vs JSON vs `labels only` vs chrome, plus length and single-word rewording effects | `docs/computer-use.md` §3 |
| `flappy_recipe_probe.py` | state serialisation (prose / JSON / bare numbers) and the question-clause effect, on the flappy question | `laya-game-playing` → `references/state-wording.md` |
| `snake_loop_probe.py` | the laya-mlx Snake policy loop ported to this engine: moves/s, call/engine latency percentiles, shield interventions, and aux answers vs the game's own ground truth | `docs/game-playing.md` |
| `route_step_probe.py` | the two call paths of step 2: the same task routed over HTTP (`/route`) and over MCP (`route_step`), asserting one schema digest and one tier from both | `docs/router-service.md` (step 2) |
| `a11y_diff_probe.py` | step 3 on real accessibility diffs: `--pairs-only` prints the size/question profile without an engine, otherwise `verify_step` is measured per question kind against its majority-class baseline | `docs/verify-step.md` |
| `setup_laya_sdk.sh` | installs the **Python SDK** (torch) path — not the ggmlc binary path the MCP actually serves | — |

Configuration is the same environment the MCP server uses: `LAYA_EXE`, `LAYA_MODEL`, and optionally
`LAYA_CUDA_GRAPH` / `LAYA_DEVICE`. Each script imports `laya_mcp_server` from the repo root, so run
them from here (or set `PYTHONPATH` to the repo root) and point `sys.path` at the same tree:

```bash
cd probes
LAYA_EXE=/path/to/laya LAYA_MODEL=/path/to/laya_multilingual_q8_0.gguf python flappy_recipe_probe.py
```

Every figure is checkpoint-specific. The upstream repo's 0.82/0.04 became 0.67/0.20 on the
multilingual Q8_0 ggmlc engine — re-measure before trusting any threshold.