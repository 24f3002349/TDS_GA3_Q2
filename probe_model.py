"""Calls the configured model the way the agent does and prints the raw outcome (API key redacted).
Usage: python probe_model.py [packages_per_call]     (reads .env if present; uses no grader checks)"""
import asyncio
import json
import os
import sys
import time

if os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())

import httpx  # noqa: E402

import decide  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else decide.LLM_CHUNK
FILLER = ("Archived training example: a different invoice was once placed on hold pending a call-back [AR-{i}-1] "
          "[AR-{i}-2] [AR-{i}-3]; this is illustrative only and does not apply. ") * int(os.environ.get("PROBE_FILL", "6"))


def package(i):
    return {"packageId": f"PROBE-{i}", "documents": [
        {"title": "Cover sheet", "text": f"Transmittal [CV-{i}] for review."},
        {"title": "Invoice", "text": f"Vendor: Probe Supplies {i}. Invoice PS-{i} for INR 12,{i}00.00."},
        {"title": "Archive", "text": FILLER.format(i=i)},
        {"title": "Reconciliation", "text": f"PO [PO-{i}] and GRN [GRN-{i}] match invoice [INV-{i}]; within the "
                                            "autonomous limit and fully reconciled."}]}


def redact(text):
    key = os.environ.get("LLM_API_KEY", "")
    return text.replace(key, "<key>") if key else text


async def main():
    print(f"base={decide.LLM_BASE_URL} models={decide.LLM_MODELS} effort={decide.LLM_REASONING or '-'} "
          f"packages/call={N} key={'set' if decide.LLM_API_KEY else 'MISSING'}")
    pkgs = [package(i) for i in range(N)]
    payload = decide.build_payload(pkgs, {"policyRevision": "PROBE"}, json_mode=True,
                                   reasoning=os.environ.get("PROBE_REASONING", "1") == "1")
    print(f"approx prompt tokens: {len(json.dumps(payload)) // 4}, max_tokens: {payload.get('max_tokens')}")
    async with httpx.AsyncClient() as client:
        for model in decide.LLM_MODELS:
            t = time.time()
            r = await client.post(f"{decide.LLM_BASE_URL}/chat/completions", json={**payload, "model": model},
                                  timeout=60, headers={"Authorization": f"Bearer {decide.LLM_API_KEY}"})
            print(f"\n{model}: HTTP {r.status_code} in {time.time() - t:.1f}s")
            if r.status_code >= 400:
                print(redact(r.text[:800]))
                print("agent would treat this as:", decide.classify(r))
                continue
            content = r.json()["choices"][0]["message"]["content"]
            try:
                decisions = decide.parse_json(content).get("decisions", [])
                print(f"parsed {len(decisions)}/{N} decisions:",
                      [(d.get("packageId"), d.get("action"), d.get("decisiveRefs")) for d in decisions])
            except Exception as e:
                print("unparseable reply:", type(e).__name__, content[:400])
            print("usage:", r.json().get("usage"))


asyncio.run(main())
