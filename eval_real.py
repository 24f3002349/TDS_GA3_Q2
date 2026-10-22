"""Score decisions on real grader packages captured from the [diag] log (realcases/labels.json, kept out of git).

Usage:  python eval_real.py            (no model: case reader / fallback path)
        python eval_real.py --model    (LLM_* settings from .env)
Checks action, the exact three evidence refs, exact facts, and that the rationale cites the refs."""
import asyncio
import json
import os
import sys

if "--model" in sys.argv and os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())
if "--model" not in sys.argv:
    os.environ["LLM_API_KEY"] = ""
    os.environ.pop("FALLBACK_API_KEY", None)

import decide  # noqa: E402

data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "realcases", "labels.json"),
                      encoding="utf-8"))


class NoStore:
    def get_decision(self, k):
        return None

    def put_decision(self, k, v):
        pass


async def main():
    pids = list(data["labels"])
    pkgs = [data["packages"][p] for p in pids]
    loop = asyncio.get_running_loop()
    results = await decide.decide_packages(pkgs, loop.time() + 600, NoStore(), {"policyRevision": "GA5-AP-2026.2"})
    bad = unsafe = 0
    for pid, r in zip(pids, results):
        want = data["labels"][pid]
        problems = []
        if r["action"] != want["action"]:
            problems.append(f"action {r['action']} != {want['action']}")
            unsafe += r["action"] == "settle_invoice"
        if [x.strip("[]") for x in r["evidenceRefs"]] != want["refs"]:
            problems.append(f"refs {r['evidenceRefs']} != {want['refs']}")
        if r["facts"] != want["facts"]:
            problems.append(f"facts {r['facts']} != {want['facts']}")
        if not (60 <= len(r["rationale"]) <= 1500) or sum(f"[{x}]" in r["rationale"] for x in want["refs"]) < 2:
            problems.append("rationale")
        bad += bool(problems)
        if problems or "-v" in sys.argv:
            print(f"{'BAD' if problems else 'OK '} {pid[-6:]} {want['action']:17s} [{r['source']}] " + "; ".join(problems))
    print(f"\n{len(pids) - bad}/{len(pids)} fully correct; unsafe settles: {unsafe}")
    if "-v" in sys.argv:
        print("\nsample rationale:", results[0]["rationale"])
    sys.exit(1 if bad else 0)


asyncio.run(main())
