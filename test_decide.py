"""Offline checks for decide.py validation (no network, no model). Usage: python test_decide.py"""
import decide

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))
    if not cond:
        failures.append(name)


PKG = {"packageId": "P1", "documents": [
    {"text": "Cover sheet for review [CV-1] [CV-2] [CV-3]."},
    {"text": "Invoice No. 4471 from Delta Foods for INR 12,000.00."},
    {"text": "Bank-detail change call-back is pending [BNK-4] for invoice 4471 [INV-4471] per control [CTL-9]."},
    {"id": "EV-7", "text": "Structured evidence object."}]}

# refs: exact package IDs only, snapped to the decisive paragraph, invented IDs dropped
check("snap to decisive paragraph", decide.resolve_refs(PKG, ["BNK-4", "inv-4471"]) == ["BNK-4", "INV-4471", "CTL-9"])
check("invented IDs dropped", decide.resolve_refs(PKG, ["FAKE-1", "[BNK-4]"]) == ["BNK-4"])
check("structured id accepted", decide.resolve_refs(PKG, ["EV-7"]) == ["EV-7"])
check("package id is not evidence", decide.resolve_refs(PKG, ["P1"]) == [])

# heuristic: "Invoice No." is not a negation, noise paragraphs skipped
h = decide.heuristic(PKG)
check("heuristic hold", h["action"] == "hold_invoice" and h["decisiveRefs"] == ["BNK-4", "INV-4471", "CTL-9"], h)
neg = {"packageId": "P2", "documents": [{"text": "Callback is not pending [A-1]; no hold applies [A-2] [A-3]."},
                                        {"text": "PO [PO-1] and GRN [GRN-1] match invoice [INV-1]; fully reconciled."}]}
check("negated cue ignored", decide.heuristic(neg)["action"] == "settle_invoice", decide.heuristic(neg))

# model decision with a hallucinated ref still cites real evidence and explains each ref
d = decide.finalize(PKG, {"action": "hold_invoice", "decisiveRefs": ["BNK-4", "INV-4471", "CTL-9"],
                          "evidence": [{"ref": "BNK-4", "shows": "the bank-detail call-back is still pending"}],
                          "reason": "Payment must wait for the call-back.", "currency": "INR", "amountText": "12,000.00"})
check("finalize refs", d["evidenceRefs"] == ["BNK-4", "INV-4471", "CTL-9"], d)
check("rationale cites every ref", all(f"[{r}]" in d["rationale"] for r in d["evidenceRefs"]), d["rationale"])
check("rationale explains", "call-back is still pending" in d["rationale"] and 60 <= len(d["rationale"]) <= 1500)
check("amount minor", d["facts"]["amountMinor"] == 1200000, d["facts"])
bad = decide.finalize(PKG, {"action": "hold_invoice", "decisiveRefs": ["NOPE-1"], "reason": "x"})
check("invalid refs replaced from package", bad["evidenceRefs"] == ["BNK-4", "INV-4471", "CTL-9"]
      and bad["source"] == "fallback-refs", bad)
check("unknown action -> open_exception", decide.finalize(PKG, {"action": "pay"})["action"] == "open_exception")
check("batch context drops packages/batchId",
      decide.batch_context({"batchId": "B", "packages": [], "policyRevision": "R"}) == {"policyRevision": "R"})

print("\nALL PASS" if not failures else f"\n{len(failures)} FAILURES: {failures}")
