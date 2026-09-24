"""Does the flappy-laya recipe hold on the local ggmlc engine?

rupeshs/flappy-laya-openvino-cpu claims two things about how Laya reads a state:
  a) a JSON dict of booleans -> ~0.9 flap probability every frame (bird flaps itself to death)
  b) two bare numbers -> no better, "it compares words, not floats"
  c) prose naming the direction -> mean 0.82 flap vs 0.04 glide
and one thing about the question:
  d) adding an "is not already rising" clause scores 0 pipes

This reruns (a)-(d) against the multilingual Q8_0 ggmlc engine with the identical question
wording, so the recipe is verified on the engine this machine actually serves.
"""

import json
import os
import sys
import time

sys.path.insert(0, "G:/dev/AI/laya_mcp")
os.environ.setdefault("LAYA_EXE", "G:/dev/AI/laya/layabin/laya.exe")
os.environ.setdefault("LAYA_MODEL", "G:/dev/AI/laya/laya_multilingual_q8_0.gguf")
os.environ.setdefault("LAYA_CUDA_GRAPH", "1")

import laya_mcp_server as srv  # noqa: E402

QUESTION = {"should_flap": {"type": "noul",
                            "instructions": "The bird is too low and must flap to climb."}}
# The variant the repo says scores 0 pipes.
QUESTION_VELOCITY = {"should_flap": {"type": "noul",
                                     "instructions": "The bird is too low and must flap to climb, "
                                                     "and it is not already rising."}}


def ask(state, questions=QUESTION):
    t0 = time.time()
    out = srv.DAEMON.call({"state": state, "questions": questions})
    ms = (time.time() - t0) * 1000
    ans = out.get("answers", {}).get("should_flap", {})
    return ans.get("noul"), ms


def cases(offset):
    """offset > 0 means the bird is BELOW the gap centre, i.e. too low."""
    low = offset > 0
    direction = "below" if low else "above"
    word = "too low" if low else "high enough"
    return {
        # exactly what the repo's describe_state() emits
        "prose (repo wording)": (
            f"The bird is flying {word} the gap. It sits {abs(offset):.3f} {direction} "
            f"the gap's center line."),
        # same content, grammatical
        "prose (grammatical)": (
            f"The bird is flying {word} for the gap. It sits {abs(offset):.3f} {direction} "
            f"the gap's center line."),
        # the shape the repo says fails
        "json booleans": json.dumps({"bird_is_below_gap_center": low,
                                     "offset_from_gap_center": offset,
                                     "bird_velocity": -0.4 if low else 0.3}),
        "json number only": json.dumps({"offset_from_gap_center": offset}),
        "bare numbers": f"{offset:.3f}, -0.4" if low else f"{offset:.3f}, 0.3",
    }


print("Question:", QUESTION["should_flap"]["instructions"])
print(f"{'state format':24s} | {'LOW  p(flap)':>14s} | {'HIGH p(flap)':>14s} | separation | ms")
print("-" * 84)
rows = {}
for offset in (0.35, -0.35):
    for name, text in cases(offset).items():
        p, ms = ask(text)
        rows.setdefault(name, {})[offset] = (p, ms, text)

for name, d in rows.items():
    pl, ml, tl = d[0.35]
    ph, mh, th = d[-0.35]
    print(f"{name:24s} | {pl:14.4f} | {ph:14.4f} | {pl - ph:+10.4f} | {(ml + mh) / 2:5.0f}")

print("\n### the question wording matters too (repo claim: velocity clause = 0 pipes)")
t_low_falling = ("The bird is flying too low the gap. It sits 0.350 below the gap's center line. "
                 "It is dropping.")
t_low_rising = ("The bird is flying too low the gap. It sits 0.350 below the gap's center line. "
                "It is already climbing.")
for label, text in (("plain question, falling", t_low_falling),
                    ("plain question, rising", t_low_rising)):
    p, ms = ask(text)
    print(f"{label:34s} plain -> {p:.4f}  ({ms:.0f} ms)")
for label, text in (("velocity question, falling", t_low_falling),
                    ("velocity question, rising", t_low_rising)):
    p, ms = ask(text, QUESTION_VELOCITY)
    print(f"{label:34s} +velocity clause -> {p:.4f}  ({ms:.0f} ms)")

srv.DAEMON.stop()