#!/usr/bin/env python
"""Phase 3: insert wait_for_signal into expert trajectories.

Expert plans are joint sequences whose ordering is implicit. Live-sim schedules
agents concurrently, so that ordering is lost and correct plans break -- in
arrange_bread_bowl/traj_000009 agent_1 carried the bowl away before agent_0
could place bread into it. Inserting explicit waits makes the ordering the plan
always assumed something the scheduler must honour.

For each cross-agent dependency (agent B acts on object X that agent A last
touched) this writes, into B's stream, immediately before B's action:

    communicate(to=A, "waiting to hear about X")   <- announcement
    wait_for_signal(from=A, about=X)               <- blocks

and guarantees A emits a release mentioning X after A's last action on it,
reusing an existing communicate where one already qualifies.

Distractor messages are deliberately NOT removed. Training needs
informative-but-unrelated traffic between a wait and its release, or the model
learns "resume on any message" -- which is the degenerate policy the harness
cannot prevent, since it wakes on any message by design.

  python insert_wait_for_signal.py --root <dir>            # dry run
  python insert_wait_for_signal.py --root <dir> --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

OBJECT_KEYS = ("object_id", "support_object_id", "reference_object_id", "receptacle_id")
SKIP = (None, "get_image", "communicate", "wait_for_signal")


def _agent(call):
    return f"agent_{call.get('robot_idx')}"


def _objects(call):
    args = call.get("args") or {}
    return [args[k] for k in OBJECT_KEYS if args.get(k)]


def _mentions(call, obj):
    msg = ((call.get("args") or {}).get("message") or "").lower()
    return obj.lower() in msg


def rewrite(calls):
    """Returns (new_calls, stats). Pure function over one trajectory."""

    stats = Counter()
    last_toucher: dict[str, str] = {}
    # first pass: locate dependencies as (index_in_calls, waiter, holder, object)
    deps = []
    for i, call in enumerate(calls):
        tool = call.get("tool")
        if tool in SKIP:
            continue
        me = _agent(call)
        for obj in _objects(call):
            holder = last_toucher.get(obj)
            if holder and holder != me:
                deps.append((i, me, holder, obj))
                break
        for obj in _objects(call):
            last_toucher[obj] = me

    if not deps:
        return calls, stats

    # second pass: build the new sequence
    dep_at = {i: (w, h, o) for i, w, h, o in deps}
    out = []
    for i, call in enumerate(calls):
        if i in dep_at:
            waiter, holder, obj = dep_at[i]
            widx = int(waiter.rsplit("_", 1)[1])
            out.append({
                "tool": "communicate", "robot_idx": widx,
                "args": {"to": holder,
                         "message": f"I need to use {obj}. Tell me when you are done "
                                    f"with it and I will proceed."},
            })
            out.append({
                "tool": "wait_for_signal", "robot_idx": widx,
                "args": {"from": holder, "about": obj},
            })
            stats["waits"] += 1
            # release: reuse an existing message from the holder that mentions
            # the object and lands before the dependent action; else synthesise
            has_release = any(
                c.get("tool") == "communicate"
                and _agent(c) == holder
                and _mentions(c, obj)
                for c in calls[:i]
            )
            if not has_release:
                hidx = int(holder.rsplit("_", 1)[1])
                out.insert(len(out) - 2, {
                    "tool": "communicate", "robot_idx": hidx,
                    "args": {"to": waiter,
                             "message": f"I am finished with {obj}; it is free for you now."},
                })
                stats["releases_synthesised"] += 1
            else:
                stats["releases_reused"] += 1
        out.append(call)
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*", "traj_*", "adapted_trajectory.json")))
    if args.limit:
        paths = paths[: args.limit]
    total = Counter()
    touched = 0
    for path in paths:
        data = json.loads(open(path, encoding="utf-8").read())
        calls = data.get("tool_calls") or []
        new_calls, stats = rewrite(calls)
        if not stats:
            continue
        touched += 1
        total.update(stats)
        if args.apply:
            data["tool_calls"] = new_calls
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)

    print(f"trajectories scanned : {len(paths)}")
    print(f"trajectories modified: {touched}")
    print(f"waits inserted       : {total['waits']}")
    print(f"releases reused      : {total['releases_reused']}")
    print(f"releases synthesised : {total['releases_synthesised']}")
    if not args.apply:
        print("\ndry run -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
