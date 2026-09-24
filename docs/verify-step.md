# `verify_step` on real accessibility diffs (step 3)

`docs/computer-use.md` §4.2 argued the loop should send a window's accessibility tree **once**, then
only the deltas: "a diff is ~200 tokens against a 12k-token window, and it fits the 768-token state
budget comfortably." This is that argument measured on the captures already sitting on this machine,
with the gold answers derived from the trees rather than written by hand.

**Verdict: the diff channel is cheap but the tool is not accurate enough to gate anything.** At
`labels` serialisation Laya answers **0.602 (62/103)** on real diffs, against a **0.569** majority-class
baseline on the largest question kind and **0.733** on the error question. A hosted reader lands at
0.650 — *not* separable from Laya (McNemar p = 0.542) at 600× the latency and 14 unanswered items
against 0. So `verify_step` ships **advisory**, with its measured accuracy and its error rate stated in
the tool description itself, and nothing in this repo gates a step on it.

## Method

* **Corpus**: 20 cua-driver `mode='ax'` captures already on this machine (4 apps, one of them 12
  captures of the same window), paired into **15 same-app consecutive-in-time pairs** — real
  before/after states of a real window, not a synthetic diff.
* **Questions, 103 per run**: up to 4 `present` (asked about lines the diff actually added and
  removed), 1 `role_present` (rotating through a fixed role shortlist by pair), 1 `error_present`,
  1 `net_added` (a closed 3-way choice). Templates: `laya_router/data/questions.verify.json`.
* **Gold is derived, never authored**: `present` is checked against the after-tree, `role_present` and
  `error_present` are properties of the after-tree, `net_added` counts the lines the model can see.
  Candidates come from the *rendered* lines, so a truncated diff never hides the evidence.
* **Baselines**: every kind is reported next to the accuracy a constant answer would get
  (`majority_baseline`). Without that column, `role_present` at 0.667 looks like a capability when its
  baseline is 0.533.
* **Answer space is closed** (`noul` or a 2–3-way choice), which is the quarantined-perception
  property from §4.1: the screen text has no channel back to a planner as an instruction.

Nothing about the screen is committed. The captures are machine-local; the results files hold kinds,
gold and predicted values, roles, and sizes — no labels. `probes/a11y_diff_probe.py --pairs-only`
prints the size/question profile without an engine, so the method is reproducible on any capture
directory.

## Serialisation decides a lot, but not enough

Identical items, identical backends; only the way the diff is written changes.

| variant | what it looks like | Laya accuracy (n=103) | `present` | `role_present` | `error_present` | `net_added` |
|---|---|---:|---:|---:|---:|---:|
| `labels` | `+ Error while saving the file` | **0.602** | 0.621 | 0.667 | 0.600 | 0.467 |
| `suffix` | `Text: Error while saving the file [appeared]` | 0.524 | 0.603 | 0.467 | 0.467 | 0.333 |
| `prefix` | `+ Text: Error while saving the file` | 0.515 | 0.534 | 0.600 | 0.467 | 0.400 |
| `prose` | `Labels that appeared on screen: …` | 0.505 | 0.483 | 0.600 | 0.467 | 0.533 |
| *majority-class baseline* | — | — | 0.569 | 0.533 | 0.733 | 0.533 |

Dropping the role and keeping only labels — the form that took a guard head from 0.03 to 0.998 in
`computer-use.md` §3 — is worth **+8.7 points** here too, and it is the default for that reason. But
the whole span is 0.505–0.602: serialisation is a real lever and not a solution. And `error_present`
stays **below its 0.733 baseline** in every variant, which is the sharpest single result in this
document: asked "does the current screen contain an error?", the honest constant answer beats the
model's reading.

## Laya vs a hosted reader

Same 103 items, same diffs, same `labels` serialisation, same `answer()` path for both backends:

| | Laya (Q8_0, local) | `qwen3.8-flash` (hosted) |
|---|---:|---:|
| overall accuracy | 0.602 (62/103) | 0.650 (67/103) |
| 95% CI | 0.505–0.691 | 0.555–0.736 |
| `present` | 0.621 | **0.793** |
| `role_present` | **0.667** | 0.333 |
| `error_present` | **0.600** | 0.267 |
| `net_added` | 0.467 | **0.800** |
| unanswered items | **0** | 14 |
| latency p50 / p99 | **37 ms / 272 ms** | 21.9 s / 90.7 s |
| paid tokens per run | 0 (local) | ~64k in |

Paired discordance: **43 both right, 19 Laya-only, 24 hosted-only, 17 neither — McNemar exact
p = 0.542**. A 4.8-point gap on 103 items is a coin flip, and the per-kind split says why they tie:
the hosted reader is much better at the questions that are *reading a label* (`present`, `net_added`)
and much worse at the two that are *properties of the whole current tree* (`role_present`,
`error_present`) — and it failed 14 items by not answering at all, which count as misses.

That is the step-1 shape again, on a harder task: **Laya is not separable from a hosted model on
accuracy here, and is 600× cheaper in wall-clock**. It is also nowhere near good enough to decide
whether a step succeeded.

## The cost is the question set, not the diff

The claim in §4.2 — a diff is ~200 tokens — is true of the diff:

| what | measured |
|---|---|
| diff chars across the 15 pairs, `labels` run | p50 **92**, p90 **1,218**, max **1,265** (~575 tokens) |
| pairs cut at the 40-line-per-side bound | **5 of 15** (the window-repaint cases; none reached the 1,600-char bound first) |
| engine input tokens, **1** question on a 1,606-char diff | **~660** |
| engine input tokens, **7** questions on the same diff | **~4,600** |
| engine input tokens, 7 questions on a 92-char diff | **1,072** (p50 over the run) |
| share of calls over the 768-token documented state budget | **95.1%** (98 of 103), p90 3,933, max 4,144 |

The encoder's cost is **questions × (state + question text)**, not state plus questions: 7 questions over
one 1,606-char diff cost ~4,600 tokens, which is ~7 × the single-question cost on the identical state —
the state is re-read per question rather than shared. The 92-char column is the same arithmetic from
the other end: with the diff almost free, seven questions still cost ~1,000 tokens, because their own
text is what is being paid for. So "fits the 768-token budget comfortably" is false for a question set
this size no matter how small the diff is. Two things follow, and both are now in the shipped tool:
**ask few questions per call**, and keep the instructions short — the amount of question text is what
the engine is paying for, and this is where a fine-tune would earn its keep.

## What ships, and what would change the verdict

* `verify_step(before, after, backend?, max_lines?, timeout_ms?)` — the MCP surface, returning typed
  answers with the probability behind each, the diff's size, `advisory: true`, and its measured
  accuracy in the payload *and* the tool description.
* `laya_router/a11y.py` — capture parsing, the diff, the four serialisations, question generation with
  derived gold, and the answer reader. `laya_router/verify.py` — the measurement harness.
* `advisory` stays the only mode. Per issue #3's acceptance criteria, gating would flip to `enforce`
  only after a measured eval, and there is no such eval for this task yet.

What would change the verdict, in order of expected value:

1. **A decision head trained on the task** (step 5 of #3). The base checkpoint is near chance on
   bespoke typed questions; nothing here suggests this task is different.
2. **A question set built for the encoder**, not for a chat model: fewer, terser questions, with the
   answer space as small as the decision allows. The token profile above says this is where the cost
   and probably some of the accuracy is.
3. **Not more serialisation tuning.** Four variants span 9.7 points and the best is still under the
   error-question baseline.

## Boundaries

* **An accessibility diff cannot see an image-embedded injection.** If the bypass is instructions
  rendered as pixels, the tree carries no text to answer about, and a plausible-typed answer is the
  worst possible outcome. This is the §6 gap, and it is why both computer-use tools return
  `advisory: true` and a `boundary` string with every call.
* **A tree-diff sees tree changes only.** An action that changes pixels without changing the
  accessibility tree (a canvas app, an animation, a video) is invisible here.
* **The diff is text from the app itself.** A hostile app can put anything it likes in a label; the
  typed answer space is what stops that text from becoming an instruction, not any judgement about it.

## Reproducing

```bash
# no engine needed: the size and question profile of whatever captures are on this machine
python probes/a11y_diff_probe.py --pairs-only

# the measurement: Laya alone, or Laya against a hosted reader
export LAYA_EXE=/path/to/laya.exe LAYA_MODEL=/path/to/laya_multilingual_q8_0.gguf
python probes/a11y_diff_probe.py --variant labels
python probes/a11y_diff_probe.py --variant prefix,suffix,prose,labels --backends \
  "laya,openai:<base_url>|<model>|<KEY_ENV>" --env-file ~/.env

# a different capture directory (any folder of cua-driver mode='ax' captures)
LAYA_A11Y_CAPTURES=/path/to/captures python probes/a11y_diff_probe.py --pairs-only
```

Results land in `laya_router/data/results/verify_*.json`: one row per question with its kind, gold and
predicted value, and the diff size — no labels, no screen text. The published runs in this document:

| file | what it holds |
|---|---|
| `verify_laya_vs_qwen38flash.json` | the headline run: Laya vs the hosted reader, `labels`, both metrics and the paired discordance |
| `verify_laya_labels.json` | Laya alone, `labels` (the serialisation default) |
| `verify_laya_suffix.json`, `verify_laya_prose.json` | the ablation arms |
| `verify_laya_vs_agnes.json` | Laya plus the `prefix` arm — written before the variant was recorded in the artifact, so the file itself says nothing about it, and the 0.515 in the table comes from this run |
| `verify_laya_vs_longcat.json` | a second `labels` run, kept because it failed |

The two `vs_agnes` / `vs_longcat` hosted arms are 103/103 failures: Agnes and LongCat both answered
`quota exhausted`. They are kept as the record that the hosted comparison was attempted on four
providers before the fifth worked — OpenAI ("no credits"), OpenRouter and Agnes ($0), LongCat ("token
quota"), MiniMax ("plan does not include the model"). A table of provider failures is not a result,
but deleting it would make the hosted arm look easier than it was.

One thing inside those two files is actively misleading and worth naming: their `paired` block reports
`p = 0.000` over an arm that answered nothing, under a `scope` string (`items both answered`) that the
harness no longer produces — it was written before #9 corrected the label to "items asked of both
backends; an unanswered item counts as a miss". A McNemar p-value between a working engine and a
provider that failed 103 of 103 calls measures the outage, not the models. The artifacts are left
exactly as written: they are the evidence of a failed arm, and patching their fields afterwards would
be worse than explaining them.