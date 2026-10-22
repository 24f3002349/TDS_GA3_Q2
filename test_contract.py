"""Contract self-test. Usage: python test_contract.py [BASE_URL]   (default http://127.0.0.1:8000/a2a/)
Uses synthetic packages only; costs no grader checks."""
import asyncio
import json
import sys
import uuid

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/a2a/").rstrip("/") + "/"
ORIGIN = BASE.split("/", 3)[0] + "//" + BASE.split("/", 3)[2]
TOKEN_A, TOKEN_B = "owner-" + uuid.uuid4().hex, "outsider-" + uuid.uuid4().hex
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
failures = []


def H(token=TOKEN_A, **extra):
    h = {"Authorization": f"Bearer {token}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json",
         "Accept": "application/a2a+json"}
    h.update(extra)
    return h


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))
    if not cond:
        failures.append(name)


PACKAGES = [
    {"packageId": "PKG-1", "documents": [
        {"title": "Cover sheet", "text": "Transmittal for review [CV-001] [CV-002] [CV-003]."},
        {"title": "Invoice", "text": "Vendor: Acme Tools Pvt Ltd. Invoice INV-7781 for INR 45,200.00."},
        {"title": "Archive", "text": "Archived training example: invoice was already paid twice [AR-1] [AR-2] [AR-3]."},
        {"title": "Reconciliation", "text": "PO [PO-7781] and goods receipt [GRN-55] match invoice [INV-7781]; amount is within the autonomous limit and fully reconciled."}]},
    {"packageId": "PKG-2", "documents": [
        {"title": "Invoice", "text": "Vendor: Blue River Logistics. Invoice BRL-220 for INR 9,80,000.00."},
        {"title": "Policy", "text": "The amount exceeds the delegated authority [POL-3.1]; PO [PO-220] and receipt [GRN-220] reconcile, so CFO approval is required."}]},
    {"packageId": "PKG-3", "documents": [
        {"title": "Invoice", "text": "Vendor: Zen Paper Co. Invoice ZP-9 for INR 12,000.00."},
        {"title": "Payments", "text": "Invoice ZP-9 was already paid on 2026-03-01 under payment [PAY-9] matching ledger [LED-9] and remittance [REM-9]."}]},
]


def batch_message(mid=None):
    return {"messageId": mid or str(uuid.uuid4()), "role": "ROLE_USER",
            "parts": [{"mediaType": MT_BATCH, "data": {"batchId": "B-" + uuid.uuid4().hex[:8],
                                                       "policyRevision": "POL-2026.10", "packages": PACKAGES}}]}


def send_body(message, ri=False):
    return {"message": message, "configuration": {"returnImmediately": ri, "historyLength": 20,
                                                  "acceptedOutputModes": [MT_PROPOSALS, MT_RECEIPTS]}}


def reorder(obj):
    if isinstance(obj, dict):
        return {k: reorder(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [reorder(x) for x in obj]
    return obj


def results_message(task, outcomes=None, **override):
    props = next(p for a in task["artifacts"] for p in a["parts"] if p["mediaType"] == MT_PROPOSALS)["data"]
    results = []
    for i, p in enumerate(props["proposals"]):
        results.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                        "outcome": (outcomes or ["ACCEPTED"] * 99)[i], "receiptNonce": "nonce-" + uuid.uuid4().hex})
    msg = {"messageId": str(uuid.uuid4()), "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER",
           "parts": [{"mediaType": MT_RESULTS, "data": {"batchId": props["batchId"], "results": results}}]}
    msg.update(override)
    return msg


async def main():
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(ORIGIN + "/.well-known/agent-card.json")
        card = r.json()
        check("card 200 json", r.status_code == 200 and "json" in r.headers["content-type"])
        check("card interface", {"url": BASE, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
              in card["supportedInterfaces"], card["supportedInterfaces"])
        check("card skill", any(s["id"] == "invoice_action_agent" and s["tags"] for s in card["skills"]))

        url = BASE + "message:send"
        r = await c.post(url, json=send_body(batch_message()), headers={"A2A-Version": "1.0", "Content-Type": "application/a2a+json"})
        check("no auth -> 401", r.status_code in (401, 403), r.status_code)
        r = await c.post(url, json=send_body(batch_message()), headers=H(**{"A2A-Version": "0.3"}))
        check("bad version -> 400", r.status_code == 400, r.status_code)
        r = await c.post(url, content=b"{}", headers=H(**{"Content-Type": "text/plain"}))
        check("bad content-type -> 4xx", r.status_code in (400, 415), r.status_code)
        r = await c.post(url, json=send_body(batch_message()), headers=H(**{"Content-Type": "application/json"}))
        check("application/json body -> 415", r.status_code == 415, r.status_code)
        check("415 error envelope", r.headers.get("content-type", "").startswith("application/problem+json")
              and r.json()["error"]["details"][0]["reason"] == "CONTENT_TYPE_NOT_SUPPORTED", r.text[:300])
        r = await c.post(url, json=send_body(batch_message()), headers={"Content-Type": "application/json"})
        check("auth checked before media type", r.status_code == 401 and "www-authenticate" in r.headers, r.status_code)
        r = await c.post(url, json=send_body(batch_message()), headers={k: v for k, v in H().items() if k != "A2A-Version"})
        check("missing version -> 400", r.status_code == 400 and "VERSION_NOT_SUPPORTED" in r.text, r.status_code)
        r = await c.get(BASE + "tasks", headers=H(**{"A2A-Version": "1.0.1"}))
        check("patch version 1.0.1 accepted", r.status_code == 200, r.status_code)
        r = await c.get(BASE + f"tasks/{uuid.uuid4()}", headers=H())
        check("404 error envelope", r.status_code == 404 and r.json()["error"]["details"][0]["reason"] == "TASK_NOT_FOUND",
              r.text[:300])

        # concurrent equivalent sends: reordered keys + changed returnImmediately
        msg = batch_message()
        bodies = [send_body(msg), send_body(reorder(msg), ri=True), send_body(msg), send_body(reorder(msg))]
        hdrs = [H(), H(**{"Content-Type": "application/a2a+json; charset=utf-8"}), H(), H()]
        rs = await asyncio.gather(*(c.post(url, json=b, headers=h) for b, h in zip(bodies, hdrs)))
        check("concurrent sends 200", all(x.status_code == 200 for x in rs), [x.status_code for x in rs])
        check("send media type", rs[0].headers["content-type"].startswith("application/a2a+json"))
        tasks = [x.json()["task"] for x in rs]
        check("same task", len({t["id"] for t in tasks}) == 1 and tasks[0] == tasks[1] == tasks[3])
        task = tasks[0]
        check("INPUT_REQUIRED", task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED", task["status"])
        check("history keeps initial", task["history"][0]["messageId"] == msg["messageId"])
        parts = [p for a in task.get("artifacts", []) for p in a["parts"]]
        check("one proposals part, no receipt", [p["mediaType"] for p in parts] == [MT_PROPOSALS], parts)
        expected = {"PKG-1": "settle_invoice", "PKG-2": "request_approval", "PKG-3": "reject_duplicate"}
        for p in parts[0]["data"]["proposals"]:
            print("   ", p["packageId"], p["action"], p["evidenceRefs"], p["facts"])
            print("     ", p["rationale"])
            check(f"rationale len {p['packageId']}", 60 <= len(p["rationale"]) <= 1500)
            check(f"actionId len {p['packageId']}", len(p["actionId"]) >= 12)
            check(f"action {p['packageId']}", p["action"] == expected[p["packageId"]], p["action"])
            pkg_text = json.dumps(next(k for k in PACKAGES if k["packageId"] == p["packageId"]))
            refs = [r.strip("[]") for r in p["evidenceRefs"]]
            check(f"evidence IDs exist {p['packageId']}", refs and all(f"[{r}]" in pkg_text for r in refs), refs)
            check(f"rationale cites evidence {p['packageId']}", all(r in p["rationale"] for r in refs))

        changed = dict(msg, parts=[{"mediaType": MT_BATCH, "data": {"batchId": "other", "packages": PACKAGES}}])
        r = await c.post(url, json=send_body(changed), headers=H())
        check("idempotency conflict 409", r.status_code == 409 and "IDEMPOTENCY_CONFLICT" in r.text, r.text)

        # isolation
        r = await c.get(BASE + f"tasks/{task['id']}", headers=H(TOKEN_B))
        check("outsider get 403/404", r.status_code in (403, 404) and task["id"] not in r.text, r.status_code)
        r = await c.post(BASE + f"tasks/{task['id']}:cancel", content=b'{}', headers=H(**{"Content-Type": "application/json"}))
        check("cancel with application/json body -> 415", r.status_code == 415, r.status_code)
        r = await c.post(BASE + f"tasks/{task['id']}:cancel", headers=H(TOKEN_B))
        check("outsider cancel 403/404", r.status_code in (403, 404), r.status_code)
        r = await c.post(url, json={"message": results_message(task)}, headers=H(TOKEN_B))
        check("outsider continue 403/404", r.status_code in (403, 404), r.status_code)
        r = await c.get(BASE + "tasks", headers=H(TOKEN_B))
        check("outsider list empty", r.status_code == 200 and r.json()["tasks"] == [], r.text[:200])
        r = await c.get(BASE + "tasks", headers=H())
        check("owner list", r.status_code == 200 and any(t["id"] == task["id"] for t in r.json()["tasks"]))

        # mismatched continuations are rejected and change nothing
        for name, override in [("context", {"contextId": "wrong"}),
                               ("task", {"taskId": str(uuid.uuid4())})]:
            r = await c.post(url, json={"message": results_message(task, **override)}, headers=H())
            check(f"wrong {name} rejected", r.status_code in (400, 404, 409), r.status_code)
        bad = results_message(task)
        bad["parts"][0]["data"]["results"][0]["action"] = "settle_invoice" if bad["parts"][0]["data"]["results"][0]["action"] != "settle_invoice" else "hold_invoice"
        r = await c.post(url, json={"message": bad}, headers=H())
        check("wrong action rejected", r.status_code == 400, r.status_code)
        r = await c.get(BASE + f"tasks/{task['id']}", headers=H())
        check("still INPUT_REQUIRED", r.json()["status"]["state"] == "TASK_STATE_INPUT_REQUIRED")

        # valid continuation, one rejected
        good = results_message(task, ["ACCEPTED", "REJECTED", "ACCEPTED"])
        r = await c.post(url, json={"message": good}, headers=H())
        done = r.json().get("task", {})
        check("COMPLETED", r.status_code == 200 and done["status"]["state"] == "TASK_STATE_COMPLETED", r.text[:300])
        rec = [p for a in done["artifacts"] for p in a["parts"] if p["mediaType"] == MT_RECEIPTS]
        check("one receipt part, accepted only", len(rec) == 1 and len(rec[0]["data"]["executions"]) == 2)
        r2 = await c.post(url, json={"message": reorder(good)}, headers=H())
        check("continuation replay identical", r2.status_code == 200 and r2.json()["task"] == done)
        r3 = await c.post(url, json=send_body(msg), headers=H())
        check("initial replay returns terminal task", r3.status_code == 200 and r3.json()["task"] == done)
        r = await c.post(BASE + f"tasks/{task['id']}:cancel", headers=H())
        check("cancel terminal -> 409", r.status_code == 409, r.status_code)
        r = await c.get(BASE + f"tasks/{task['id']}", headers=H())
        check("terminal unchanged", r.json() == done)

        # cancel vs receipt race
        for i in range(3):
            t = (await c.post(url, json=send_body(batch_message()), headers=H())).json()["task"]
            ra, rb = await asyncio.gather(c.post(url, json={"message": results_message(t)}, headers=H()),
                                          c.post(BASE + f"tasks/{t['id']}:cancel", headers=H()))
            final = (await c.get(BASE + f"tasks/{t['id']}", headers=H())).json()
            codes = sorted([ra.status_code, rb.status_code])
            has_receipt = any(p["mediaType"] == MT_RECEIPTS for a in final.get("artifacts", []) for p in a["parts"])
            st = final["status"]["state"]
            ok = codes == [200, 409] and ((st == "TASK_STATE_COMPLETED") == has_receipt) and st in (
                "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED")
            check(f"race {i} ({st})", ok, (codes, st, has_receipt))

    print("\nALL PASS" if not failures else f"\n{len(failures)} FAILURES: {failures}")


asyncio.run(main())
