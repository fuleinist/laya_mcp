#!/usr/bin/env python
"""Does the laya-mlx Snake policy loop hold on this repo's ggmlc daemon?

Evidence for `docs/game-playing.md`. Upstream (mizorewww/laya-mlx) runs Laya in-process on MLX
against a deterministic cycle-safety planner; this probe runs the *same* three questions per
frame, with the *same* compact wording and the *same* shield semantics, through the daemon this
repo actually serves (one JSON line per call, strict FIFO) — headless, any platform, no MLX.

What is measured: sustained moves/second, daemon-call and engine latency percentiles, shield
interventions, score, and — because Snake has ground truth — the aux answers against what the
game itself computes (food reachability; safe-route availability).

Game rules, move classification and questionnaire wording are adapted from mizorewww/laya-mlx
(Apache-2.0): `laya_mlx/snake/game.py` and the "compact" prompt in `laya_mlx/snake/policy.py`.
Keep the wording identical when comparing numbers — the wording is part of the measurement.

    cd probes
    LAYA_EXE=... LAYA_MODEL=... python snake_loop_probe.py --moves 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

REPO_ROOT = os.environ.get("LAYA_MCP_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("LAYA_EXE", "G:/dev/AI/laya/layabin/laya.exe")
os.environ.setdefault("LAYA_MODEL", "G:/dev/AI/laya/laya_multilingual_q8_0.gguf")
os.environ.setdefault("LAYA_CUDA_GRAPH", "1")

import laya_mcp_server as srv  # noqa: E402


# ---------------------------------------------------------------------------
# Game rules: adapted from mizorewww/laya-mlx (Apache-2.0) laya_mlx/snake/game.py
# ---------------------------------------------------------------------------

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def hamiltonian_cycle(width, height):
    """Visit each square once with adjacent steps, including the closing edge."""
    if min(width, height) < 4 or (width % 2 and height % 2):
        raise ValueError("Board dimensions must be >= 4, with at least one even dimension")
    if height % 2:
        return [(y, x) for x, y in hamiltonian_cycle(height, width)]
    path = [(0, 0)]
    for y in range(height):
        xs = range(1, width) if y % 2 == 0 else range(width - 1, 0, -1)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(height - 1, 0, -1))
    return path


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool
    safe: bool
    advance: int
    reason: str
    eats: bool


class SnakeGame:
    """Deterministic rules and a separately identified cycle safety planner (port)."""

    def __init__(self, width=24, height=16, seed=7, initial_length=6):
        self.width, self.height, self.seed = width, height, seed
        self.cycle = hamiltonian_cycle(width, height)
        self.indices = {cell: index for index, cell in enumerate(self.cycle)}
        self.capacity = width * height
        if not 2 <= initial_length < self.capacity:
            raise ValueError("Initial length must be >= 2 and smaller than the board")
        self.initial_length = initial_length
        self.rng = random.Random(seed)
        start = self.indices[(width // 2, height // 2)]
        self.body = deque(self.cycle[(start - i) % self.capacity] for i in range(initial_length))
        self.score = self.ticks = 0
        self.alive, self.won = True, False
        self.death_reason = None
        self.food = self._spawn_food()

    @property
    def head(self):
        return self.body[0]

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [cell for cell in self.cycle if cell not in occupied]
        return self.rng.choice(empty) if empty else None

    def target(self, direction):
        dx, dy = VECTORS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal_reason(self, direction):
        x, y = cell = self.target(direction)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "wall"
        if cell == self.body[1]:
            return "reverse"
        occupied = set(self.body)
        if cell != self.food:
            occupied.remove(self.body[-1])  # The tail moves on a non-growing step.
        return "body" if cell in occupied else "legal"

    def moves(self):
        if not self.alive or self.won:
            return []
        head_index = self.indices[self.head]
        tail_distance = (self.indices[self.body[-1]] - head_index) % self.capacity
        food_distance = (self.indices[self.food] - head_index) % self.capacity
        moves = []
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            legal = reason == "legal"
            target = self.target(direction)
            advance = (self.indices.get(target, head_index) - head_index) % self.capacity
            eats = target == self.food
            safe = legal
            if safe and (advance > tail_distance or (advance == tail_distance and eats)):
                safe, reason = False, "would cross the tail"
            if safe and (advance == 0 or advance > food_distance):
                safe, reason = False, "would skip the food on the safe route"
            moves.append(MoveInfo(direction, legal, safe, advance, reason, eats))
        return moves

    def food_reachability(self):
        """Current empty-cell connectivity; the occupied tail is not treated as empty."""
        blocked = set(self.body) - {self.head}
        visited = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in VECTORS.values():
                cell = x + dx, y + dy
                if (0 <= cell[0] < self.width and 0 <= cell[1] < self.height
                        and cell not in blocked and cell not in visited):
                    visited.add(cell)
                    queue.append(cell)
        return self.food in visited, len(visited)

    def step(self, direction):
        if not self.alive or self.won:
            raise RuntimeError("Cannot step a finished game")
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        self.ticks += 1
        reason = self.legal_reason(direction)
        if reason != "legal":
            self.alive, self.death_reason = False, reason
            return False
        target = self.target(direction)
        self.body.appendleft(target)
        if target == self.food:
            self.score += 1
            if len(self.body) == self.capacity:
                self.won, self.food = True, None
            else:
                self.food = self._spawn_food()
            return True
        self.body.pop()
        return False

    def cycle_order_valid(self):
        indices = [self.indices[cell] for cell in reversed(self.body)]
        distances = [(b - a) % self.capacity for a, b in zip(indices, indices[1:])]
        return all(d > 0 for d in distances) and sum(distances) < self.capacity


# ---------------------------------------------------------------------------
# The questionnaire: the demo's "compact" prompt, verbatim (policy.py 169-183)
# ---------------------------------------------------------------------------

def build_questions(moves, preferred):
    criteria = {}
    for move in moves:
        if not move.legal:
            criteria[move.direction] = "Blocked. Collision."
        elif not move.safe:
            criteria[move.direction] = "Unsafe. Traps the snake."
        elif move.eats:
            criteria[move.direction] = "Safe. Eat food now. Best."
        elif move.direction == preferred:
            criteria[move.direction] = "Safe. Best route to food."
        else:
            criteria[move.direction] = "Safe. Slower route."
    return {
        "move": {"type": "choice", "instructions": "Choose the best safe move toward food.",
                 "criteria": criteria},
        "risk": {"type": "noul", "instructions": "Is a safe route available?"},
        "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
    }


def decide(game, guarded=True, timeout_ms=None):
    """One frame: planner features -> state + 3 questions -> probabilities + shield decision."""
    moves = game.moves()
    safe = [m for m in moves if m.safe]
    if not safe and guarded:
        raise RuntimeError("no safe move available (the shield never lets this happen)")
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    reachable, _space = game.food_reachability()
    state = (f"Safe route: {'yes' if safe else 'no'}. "
             f"Food reachable through empty cells: {'yes' if reachable else 'no'}.")
    questions = build_questions(moves, preferred)

    started = time.perf_counter()
    out = srv.DAEMON.call({"state": state, "questions": questions}, timeout_ms)
    wall_ms = (time.perf_counter() - started) * 1000

    answers = out.get("answers") or {}
    if "probabilities" not in (answers.get("move") or {}):
        raise RuntimeError(f"unexpected daemon answer: {json.dumps(out)[:400]}")
    probs = {d: float(answers["move"]["probabilities"].get(d, 0.0)) for d in DIRECTIONS}
    proposed = max(DIRECTIONS, key=lambda d: probs[d])
    allowed = [m.direction for m in safe]
    if proposed in allowed or not guarded:
        executed = proposed
    else:
        executed = max(allowed, key=lambda d: probs[d])
    usage = out.get("usage") or {}
    return {
        "probs": probs,
        "proposed": proposed,
        "executed": executed,
        "intervened": proposed != executed,
        "preferred": preferred,
        "risk": float((answers.get("risk") or {}).get("noul", float("nan"))),
        "risk_truth": bool(safe),
        "food": float((answers.get("food") or {}).get("noul", float("nan"))),
        "food_truth": reachable,
        "wall_ms": wall_ms,
        "engine_ms": usage.get("latency_ms"),
        "input_tokens": usage.get("input_tokens"),
    }


def pct(values, p):
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100)
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def run(args):
    guarded = not args.unassisted
    game = SnakeGame(args.width, args.height, args.seed, args.initial_length)
    warm = SnakeGame(args.width, args.height, args.seed + 10000, args.initial_length)
    cold_ms = None
    for _ in range(6):  # warmup; the demo also does an unmeasured warmup
        d = decide(warm, guarded, args.timeout_ms)
        if cold_ms is None:
            cold_ms = d["wall_ms"]
        warm.step(d["executed"])

    wall, engine, tokens = [], [], []
    food_probs, food_truth, risk_probs, risk_truth = [], [], [], []
    interventions = []
    frames = []
    violations = []
    agree_preferred = 0
    last_food = 0
    steps = 0
    started = time.perf_counter()
    for _ in range(args.moves):
        if not game.alive or game.won:
            break
        d = decide(game, guarded, args.timeout_ms)
        steps += 1
        wall.append(d["wall_ms"])
        if d["engine_ms"] is not None:
            engine.append(float(d["engine_ms"]))
        if d["input_tokens"] is not None:
            tokens.append(int(d["input_tokens"]))
        if d["intervened"]:
            interventions.append(f"{d['proposed']}->{d['executed']}")
        agree_preferred += d["executed"] == d["preferred"]
        food_probs.append(d["food"])
        food_truth.append(d["food_truth"])
        risk_probs.append(d["risk"])
        risk_truth.append(d["risk_truth"])
        frames.append({
            "tick": game.ticks,
            "proposed": d["proposed"],
            "executed": d["executed"],
            "intervened": d["intervened"],
            "probabilities": d["probs"],
            "risk_p": d["risk"],
            "food_p": d["food"],
            "food_truth": d["food_truth"],
            "wall_ms": round(d["wall_ms"], 3),
            "engine_ms": d["engine_ms"],
            "input_tokens": d["input_tokens"],
        })
        ate = game.step(d["executed"])
        if ate:
            last_food = game.ticks
        if guarded and game.alive and not game.cycle_order_valid():
            violations.append(f"tick {game.ticks}: cycle order broken")
            break
        if guarded and game.ticks - last_food > game.capacity:
            violations.append(f"tick {game.ticks}: no food progress within a full cycle")
            break
        if not game.alive or game.won:
            break
    elapsed = time.perf_counter() - started

    yes_t = [p for p, t in zip(food_probs, food_truth) if t]
    yes_f = [p for p, t in zip(food_probs, food_truth) if not t]
    food_acc = (sum((p >= 0.5) == t for p, t in zip(food_probs, food_truth)) / len(food_truth)
                if food_truth else float("nan"))
    risk_acc = (sum((p >= 0.5) for p in risk_probs) / len(risk_probs) if risk_probs
                else float("nan"))  # "safe route available" is True on every shielded frame

    summary = {
        "engine": {k: srv.DAEMON.health().get(k) for k in ("exe", "model", "family", "device", "cuda_graph")},
        "guarded": guarded,
        "seed": args.seed, "requested_moves": args.moves, "steps": steps,
        "seconds": round(elapsed, 2),
        "moves_per_second": round(steps / elapsed, 2) if elapsed else None,
        "questions_per_second": round(steps * 3 / elapsed, 1) if elapsed else None,
        "cold_start_ms": round(cold_ms, 1) if cold_ms is not None else None,
        "call_ms": {"p50": pct(wall, 50), "p95": pct(wall, 95), "p99": pct(wall, 99)},
        "engine_ms": {"n": len(engine), "p50": pct(engine, 50), "p95": pct(engine, 95)},
        "input_tokens_mean": mean(tokens),
        "score": game.score, "length": len(game.body),
        "alive": game.alive, "won": game.won, "death_reason": game.death_reason,
        "interventions": len(interventions), "intervention_pairs": interventions[:10],
        "executed_equals_planner_best": (agree_preferred / steps) if steps else None,
        "food_reachability": {
            "n": len(food_truth), "truth_yes": sum(food_truth),
            "answer_accuracy": food_acc,
            "mean_p_when_reachable": mean(yes_t), "mean_p_when_not": mean(yes_f),
        },
        "risk_safe_route": {
            "n": len(risk_probs), "truth_yes": sum(risk_truth),
            "answer_accuracy": (sum((p >= 0.5) == t for p, t in zip(risk_probs, risk_truth))
                                / len(risk_truth) if risk_truth else float("nan")),
            "answer_yes_fraction": risk_acc,
        },
        "violations": violations,
        "frames": frames,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moves", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--unassisted", action="store_true",
                        help="execute raw top-1 without the safety shield")
    parser.add_argument("--out", default="", help="write {summary, frames} JSON here")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    finally:
        srv.DAEMON.stop()

    eng = summary["engine"]
    print("=== snake loop vs ggmlc daemon (policy ported from laya-mlx) ===")
    print(f"engine  : exe={eng['exe']} model={eng['model']} device={eng['device']} "
          f"cuda_graph={eng['cuda_graph']}")
    print(f"run     : seed={summary['seed']} steps={summary['steps']} "
          f"shield={'OFF (unassisted)' if not summary['guarded'] else 'on'} "
          f"score={summary['score']} length={summary['length']} alive={summary['alive']} "
          f"won={summary['won']}")
    print(f"speed   : {summary['moves_per_second']}/s uncapped | call p50/p95/p99 "
          f"{summary['call_ms']['p50']:.1f}/{summary['call_ms']['p95']:.1f}/"
          f"{summary['call_ms']['p99']:.1f} ms | engine p50 {summary['engine_ms']['p50']:.1f} ms")
    print(f"cost    : mean {summary['input_tokens_mean']:.1f} input tokens/frame "
          f"({summary['questions_per_second']} questions/s) | cold start "
          f"{summary['cold_start_ms']} ms")
    label = "shield" if summary["guarded"] else "raw   "
    print(f"{label}  : {summary['interventions']} interventions {summary['intervention_pairs']} | "
          f"executed == planner-best on {summary['executed_equals_planner_best']:.1%} of frames")
    fr = summary["food_reachability"]
    print(f"food    : ground truth yes={fr['truth_yes']}/{fr['n']} | answer accuracy "
          f"{fr['answer_accuracy']:.3f} | mean P(food) when reachable {fr['mean_p_when_reachable']:.3f} "
          f"vs not {fr['mean_p_when_not']:.3f}")
    rr = summary["risk_safe_route"]
    print(f"risk    : ground truth yes={rr['truth_yes']}/{rr['n']} | answer accuracy "
          f"{rr['answer_accuracy']:.3f} | yes-fraction {rr['answer_yes_fraction']:.3f}")
    print(f"invariant violations: {summary['violations'] or 'none'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())