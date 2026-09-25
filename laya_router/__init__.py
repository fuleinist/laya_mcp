"""laya-router — a two-tier step router with one question schema and two backends.

Step 1 of the computer-use integration (issue #3): route a step's *task description* to a tier,
before any screen integration exists. The point is to settle whether the local Laya System-1
checkpoint earns a seat on the critical path, with labels and paired metrics rather than an
impression.

Design constraints carried in from `docs/computer-use-api.md`:

* **Two tiers, not three.** The upstream study measured Laya's middle-tier recall at 0.13 — the
  base checkpoint cannot hold a middle tier, so the choice here is binary and the middle becomes
  an escalation rather than a label.
* **One schema, two backends.** The questions, their instructions and their option descriptions
  live in `laya_router/data/questions.json`; the Laya backend sends that object verbatim and the
  OpenAI-compatible backend renders the same object into its prompt. Rewording the options is
  therefore a one-file change — which matters, because wording drove 21 accuracy points upstream.
* **Nothing is gated on this yet.** `route_step` ships advisory; see
  `docs/router-service.md` for the measured accuracy that decision rests on.
"""

__version__ = "0.1.0"