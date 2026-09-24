"""Step 3 evidence: `verify_step` measured on real accessibility diffs.

The captures this reads are machine-local — a directory of cua-driver `mode='ax'` captures written by
whatever was on screen — so nothing about the screen is committed. What is committed is the *method*:
this probe turns any capture directory into a paired, question-level measurement whose gold answers
are derived from the two trees, and `laya_router/data/results/verify_*.json` holds the outcome as
kinds, gold/predicted values and sizes only (no labels).

Two modes, because the interesting failure and the interesting number are different:

    # no engine needed: what the diffs look like, and what each one would cost
    python probes/a11y_diff_probe.py --pairs-only

    # the measurement itself (needs LAYA_EXE + LAYA_MODEL)
    python probes/a11y_diff_probe.py
    python probes/a11y_diff_probe.py --variant labels,prefix --backends "laya,openai:<base_url>|<model>|<KEY_ENV>"

`--pairs-only` is the honest way to run this on a machine with no engine: it prints the diff size and
question profile, which is what decides whether a diff fits the window at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laya_router import a11y, backends, verify  # noqa: E402


def pairs_only(capture_dir: str | None, variant: str, max_chars: int) -> int:
    captures = a11y.load_capture_dir(capture_dir)
    pairs = a11y.pair_captures(captures)
    print(f"captures: {len(captures)} -> {len(pairs)} pairs of the same app, consecutive in time")
    if not pairs:
        print("no pairs: a pair needs two captures of one app")
        return 2
    work = verify.build_items(pairs, max_chars=max_chars, variant=variant)
    print(f"\n{'app':24s} {'before':>7s} {'after':>7s} {'lines':>5s} {'chars':>6s} "
          f"{'~tokens':>7s} {'qs':>3s} trunc")
    for entry in work:
        d = entry["diff"]
        print(f"{entry['app'][:24]:24s} {d['elements_before']:7d} {d['elements_after']:7d} "
              f"{d['lines']:5d} {d['chars']:6d} {d['approx_tokens']:7d} {len(entry['items']):3d} "
              f"{d['truncated']}")
    chars = sorted(e["diff"]["chars"] for e in work)
    print(f"\ndiff chars: p50 {chars[len(chars) // 2]}, max {chars[-1]}; "
          f"variants available: {', '.join(a11y.DIFF_VARIANTS)}")
    print("a question set is the expensive part, not the diff: the encoder pays questions x "
          "(state + question text) (docs/verify-step.md)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", default=None,
                        help=f"capture directory (default {a11y.DEFAULT_CAPTURES})")
    parser.add_argument("--pairs-only", action="store_true", help="no engine: the size profile only")
    parser.add_argument("--variant", default="labels", choices=list(a11y.DIFF_VARIANTS))
    parser.add_argument("--backends", default="laya",
                        help="comma-separated specs for the measurement (default laya)")
    parser.add_argument("--env-file", default=None, help="KEY=VALUE file to load, never echoed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.pairs_only:
        return pairs_only(args.captures, args.variant, args.max_chars)

    forwarded = ["--variant", args.variant, "--backends", args.backends,
                 "--max-chars", str(args.max_chars)]
    if args.captures:
        forwarded += ["--captures", args.captures]
    if args.env_file:
        forwarded += ["--env-file", args.env_file]
    if args.limit:
        forwarded += ["--limit", str(args.limit)]
    if args.out:
        forwarded += ["--out", args.out]
    return verify.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())