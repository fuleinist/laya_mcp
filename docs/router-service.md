# Step 1: the two-tier router service, and what it measured

Step 1 of the [computer-use integration](computer-use-api.md#6-build-order) is the cheap, honest
one: route a step to a tier *before* any screen integration exists, so the question "does the local
checkpoint earn a seat on the critical path?" gets an answer with numbers instead of an opinion.

What shipped: a router service (`laya_router/`), one shared question schema, two backends behind it,
a labelled corpus of real steps from this machine, and a paired evaluation.

```
caller (Hermes | OpenClaw | curl)
   |
   +-- GET  /health      backends, schema version + digest, uptime, calls
   +-- GET  /questions   the schema both backends answer, and how it renders for a chat model
   +-- POST /route       {task, context?, backend?} -> tier, needs_tools, sensitive, advisory
   +-- GET  /route?task= same decision, for a browser or a one-liner
        |
        +-- backend laya       local ggmlc daemon, typed questions, ~16 ms
        +-- backend frontier   any OpenAI-compatible /chat/completions endpoint
```

Run it:

```bash
python -m laya_router.service --port 8760 \
  --frontier 'openai:<base_url>|<model>|<KEY_ENV>' --env-file ~/.config/keys.env
curl 'http://127.0.0.1:8760/route?task=Cut+a+release+and+publish+the+artifacts'
```

## Two tiers, not three — and why the design says so

The upstream study measured Laya's middle-tier recall at **0.13** (7 of 56 `medium` requests), and
GPT-5 nano's failure was the mirror image (41 of 61 `powerful` requests sent to `medium`). A
three-tier router is therefore not supported by the base checkpoint, and `load_schema()` **refuses**
a schema that offers three tier options rather than letting a plausible-looking config lose quietly
on the middle class. The middle belongs in an escalation decision — "T1 was not confident, or the
diff did not confirm it → go to T2" — not in a label the model has to predict.

## One schema, both backends

`laya_router/data/questions.json` holds the questions, their instructions and each option's
description. The Laya backend sends that object **verbatim** (the daemon takes JSON questions); the
chat backend gets the same object rendered into a prompt by `questions.render_prompt()`. Both answer
`tier` (choice, two options), `needs_tools` and `sensitive` (noul). Rewording the options is a data
edit — deliberately, because wording alone swung accuracy 21 points upstream.

Two schema versions are committed: `questions.json` (`router-v1`) and `questions.examples.json`
(`router-v1e`), which says the same thing with concrete examples. The ablation between them is
below, because "the wording is most of the accuracy" is exactly the kind of claim that should be
re-measured on the checkpoint in front of you.

## The corpus

`laya_router/data/steps.jsonl` — 73 real steps: 47 commit subjects from ten of this machine's
repositories, 9 typed verification questions a computer-use loop asks, 6 open issues, 4 pull
requests, 4 scheduled cron jobs, and 3 review/planning questions from this repository's own step
logs. Every line carries gold labels for the same three questions, and a `provenance` field naming
the artifact. `laya_router/data/CORPUS.md` records the labelling rubric, the judgement calls, and
the limits — including that these are one labeller's labels, so a few points of difference is noise.

## Results

Backends: `laya` is `laya_multilingual_q8_0.gguf` through the ggmlc daemon on an RTX 3090;
`frontier` is a hosted chat model answering the identical schema. Both runs below used the same
corpus digest (`1a4479f4…`) and schema digest (`e85eb281…`); the results JSON is committed under
`laya_router/data/results/`.

### `router-v1` — Laya vs `agnes-2.5-flash`

| backend | tier acc | 95% CI | economy recall | frontier recall | p50 ms | p99 ms | tier ECE | failures |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| laya | 0.616 | [0.502, 0.719] | 0.289 | **0.971** | **27** | **75** | 0.148 | 0 |
| frontier | 0.699 | [0.586, 0.792] | 0.789 | 0.600 | 3,219 | 12,837 | 0.092 | 8 |

| backend | needs_tools acc | precision / recall | sensitive acc | precision / recall |
|---|---:|---|---:|---|
| laya | 0.644 | 0.950 / 0.613 | 0.699 | 0.208 / 0.625 |
| frontier | 0.781 | 0.914 / 0.855 | 0.863 | 1.000 / 0.750 |

Paired, on the tier decision:

* all 73 items — both right 30, **Laya only 15, frontier only 21**, neither 7; exact McNemar **p = 0.405**
* the 65 items both answered — both right 30, Laya only 12, frontier only 21, neither 2; **p = 0.163**

Tokens: Laya 23,480 in / **0 out**; the chat backend 41,171 in / 15,045 out.

A second baseline (`openai/gpt-5-mini`, same corpus and schema, quota-limited) landed on the same
accuracy and the same discordant split — 0.616 vs 0.699, McNemar p = 0.405. That is the same
"tie at 35× lower latency" pattern the upstream study found, reproduced on a different pair of
models, which is the sort of thing worth noticing rather than averaging.

### What the numbers say

1. **Not separated on accuracy, separated on everything else.** 15 vs 21 discordant items is not a
   difference (p = 0.405 with all items, p = 0.163 without the failed calls). What *is* unambiguous
   is the cost side: 27 ms p50 against 3,219 ms, no tokens out, no API bill, and no quota failures
   (the chat backend lost 8 of 73 calls to rate limits and credit checks; Laya lost none).
2. **Laya's error direction is the safe one.** Economy recall 0.289 against frontier recall 0.971:
   when Laya is wrong it sends the step *up* to the expensive tier, it does not skip work that
   needed a frontier model. For a router whose job is to save money without dropping quality, that
   asymmetry is the whole argument, and it is visible only because the confusion is reported per
   class rather than as one accuracy figure.
3. **Bookkeeping flags are where Laya is weakest.** `needs_tools` 0.644 and `sensitive` 0.699, with
   `sensitive` precision 0.208 — of 8 gold-sensitive steps it caught 5, while flagging 24 in total
   (19 false positives). Precision is the number that matters for a gate: low precision means
   interruptions, high recall means fewer escapes. **Do not wire `sensitive` to a blocking gate as
   shipped.** Advisory logging only.
4. **Calibration is not there yet either.** Tier ECE 0.148 (frontier 0.092) — and the daemon's own
   `confidence` field cannot be used for it: on a two-option `choice` this checkpoint reported
   `confidence = 0.0033` while its options sat at 0.466/0.534. The eval scores calibration from
   `probabilities`, and the quirk is why both fields are kept in `Decision`.

### `router-v1e` — the wording ablation

Identical corpus, identical labels, identical backends; the only change is that `tier`'s two options
now carry concrete examples ("bump a dependency version", "design a new service whose interfaces are
not decided") instead of an abstract rule.

| backend | tier acc | 95% CI | economy recall | frontier recall | p50 ms | tier ECE | failures |
|---|---:|---|---:|---:|---:|---:|---:|
| laya | 0.466 | [0.356, 0.579] | 0.289 | 0.657 | 27 | 0.191 | 0 |
| frontier | 0.726 | [0.614, 0.815] | 0.947 | 0.486 | 2,956 | 0.184 | 0 |

Paired: both right 18, **Laya only 16, frontier only 35**, neither 4 — exact McNemar **p = 0.011**.

| wording | laya | frontier | gap | McNemar |
|---|---:|---:|---:|---:|
| `router-v1` (abstract rule) | 0.616 | 0.699 | **+8.2 pts** to frontier | p = 0.405 (not separated) |
| `router-v1e` (concrete examples) | 0.466 | 0.726 | **+26.0 pts** to frontier | p = 0.011 (separated) |

This is the upstream study's "the wording is most of the accuracy" reproduced in **magnitude** — 15
and 21 points of movement from a data edit — and **not** in direction. Concrete examples helped the
chat model (+2.7 pts, inside its interval) and cost Laya 15 points. The mechanism is plausible and
worth stating as a hypothesis rather than a finding: options share a fixed token budget in the
question head, and longer criteria are what a small encoder spends it on — consistent with Laya's
frontier recall collapsing from 0.971 to 0.657 while its economy recall stayed at 0.289. Whatever
the cause, the operational rule is the one the docs already state: **re-measure the wording on your
own checkpoint before adopting a schema**, and treat "examples make it better" as unverified.

## Verdict for step 1

**Laya earns its seat as an advisory router and a cost saver, not as a gate — and whether it is
"tied" with a hosted model depends on the wording you pick.** Under `router-v1` the two were not
separated (p = 0.405); under `router-v1e` they were, by 26 points in the hosted model's favour
(p = 0.011). What holds across both schemas is the cost side: ~2 orders of magnitude less latency,
zero output tokens, no API bill, no quota failures. Three limits ship with it:

* **The tier question is wording-sensitive on this checkpoint**: 15 points of movement from a data
  edit that helped the other backend. Ship `router-v1`, and re-measure before adopting any rewrite.
* `sensitive` is not accurate enough to block on (precision 0.208). It logs.
* `tier` over-escalates: 27 of the 38 economy steps in this corpus were sent to the frontier tier
  (economy recall 0.289). That is the safe direction, but it is also the reason the label set needs
  fine-tuning before the router can claim a saving rather than a tax.

Everything `/route` returns carries `"advisory": true`, and every reply restates the boundary:
**a text-channel decision cannot see instructions rendered into an image on screen**, so a router
like this is one layer, never the defence. That is the same boundary as
`computer-use.md` §6, restated at the point of use because a caller reading `sensitive: true` might
otherwise assume the screen was analysed.

## Reproducing

```bash
export LAYA_EXE=/path/to/laya LAYA_MODEL=/path/to/laya_multilingual_q8_0.gguf
python -m laya_router.eval \
  --backends "laya,openai:<base_url>|<model>|<KEY_ENV>" \
  --env-file ~/.config/keys.env \
  --out laya_router/data/results/local.json
```

Useful flags: `--schema` (pick a wording variant), `--limit`, `--sleep` (rate limits), `--keep-raw`
(store each backend's raw reply), `--resume <results.json>` (keep the answered predictions and retry
only the failures — a hosted backend that rate-limits mid-run is the normal case, not an exception).

Tests for the service, the schema, the metrics and the backends run with no model, no network and no
GPU: `pytest -q tests/test_router.py`.