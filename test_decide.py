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
check("invented IDs dropped, lone real ref completed to its paragraph",
      decide.resolve_refs(PKG, ["FAKE-1", "[BNK-4]"]) == ["BNK-4", "INV-4471", "CTL-9"])
check("lone cover-sheet ref not completed", decide.resolve_refs(PKG, ["CV-1"]) == ["CV-1"])
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
check("invalid refs replaced by the decisive paragraph", bad["evidenceRefs"] == ["BNK-4", "INV-4471", "CTL-9"], bad)
check("unknown action -> case reader decides (never settle)",
      decide.finalize(PKG, {"action": "pay"})["action"] == "hold_invoice")
check("batch context drops packages/batchId",
      decide.batch_context({"batchId": "B", "packages": [], "policyRevision": "R"}) == {"policyRevision": "R"})

# settle guard: a live risk record anywhere blocks settle_invoice; noise, negation, resolution and other invoices do not
def pkg(*texts):
    return {"packageId": "G", "documents": [{"text": t} for t in texts]}


RECON = "PO [PO-1] and GRN [GRN-1] match invoice [INV-1]; fully reconciled and within the autonomous limit."
SETTLE = {"action": "settle_invoice", "decisiveRefs": ["PO-1", "GRN-1", "INV-1"], "invoiceNumber": "INV-1"}
g = decide.finalize(pkg("Invoice INV-1 was already paid on 2026-02-01 via payment [PAY-1] [LED-1] [REM-1].", RECON), SETTLE)
check("guard: duplicate blocks settle", g["action"] == "reject_duplicate" and g["evidenceRefs"] == ["PAY-1", "LED-1", "REM-1"], g)
g = decide.finalize(pkg("Bank-detail call-back is pending for this vendor [BNK-2].", RECON), SETTLE)
check("guard: pending check blocks settle", g["action"] == "hold_invoice" and "Safety check" in g["rationale"], g)
for name, text in [("archive example", "Archived training example: invoice already paid [AR-1] [AR-2] [AR-3]."),
                   ("negated", "No duplicate was found and the call-back is not pending [N-1]."),
                   ("resolved", "The bank-detail call-back is pending verification, now completed [C-1]."),
                   ("other invoice paid", "Invoice INV-0 was already paid on 2026-01-05 [P-0] [L-0] [R-0].")]:
    g = decide.finalize(pkg(text, RECON), SETTLE)
    check(f"guard: {name} keeps settle", g["action"] == "settle_invoice", (g["action"], g["rationale"]))
g = decide.finalize(pkg("Bank-detail call-back is pending [BNK-2].", RECON), {**SETTLE, "action": "hold_invoice"})
check("guard: non-settle untouched", g["action"] == "hold_invoice" and g["source"] == "model", g)

# replays of real gpt-oss-20b answers to the contract-test packages
PKG1 = pkg("Transmittal for review [CV-001] [CV-002] [CV-003].",
           "Vendor: Acme Tools Pvt Ltd. Invoice INV-7781 for INR 45,200.00.",
           "Archived training example: invoice was already paid twice [AR-1] [AR-2] [AR-3].",
           "PO [PO-7781] and goods receipt [GRN-55] match invoice [INV-7781]; amount is within the autonomous limit "
           "and fully reconciled.")
g = decide.finalize(PKG1, {"action": "reject_duplicate", "decisiveRefs": ["AR-1", "AR-2", "AR-3"],
                           "invoiceNumber": "INV-7781", "reason": "Archived records show it was paid."})
check("noise guard: archive-based duplicate -> settle", g["action"] == "settle_invoice"
      and g["evidenceRefs"] == ["PO-7781", "GRN-55", "INV-7781"] and g["source"] == "noise-guard", g)
check("noise guard rationale cites refs", all(f"[{r}]" in g["rationale"] for r in g["evidenceRefs"]), g["rationale"])
g = decide.finalize(PKG1, {"action": "settle_invoice", "decisiveRefs": ["PO-7781", "GRN-55", "INV-7781"],
                           "invoiceNumber": "INV-7781"})
check("correct 120b answer untouched", g["action"] == "settle_invoice" and g["source"] == "model", g)
PKG2 = pkg("Vendor: Blue River Logistics. Invoice BRL-220 for INR 9,80,000.00.",
           "The amount exceeds the delegated authority [POL-3.1]; PO [PO-220] and receipt [GRN-220] reconcile, so CFO "
           "approval is required.")
g = decide.finalize(PKG2, {"action": "request_approval", "decisiveRefs": ["POL-3.1"]})
check("single ref completed to three", g["action"] == "request_approval"
      and g["evidenceRefs"] == ["POL-3.1", "PO-220", "GRN-220"], g)
PKG3 = pkg("Vendor: Zen Paper Co. Invoice ZP-9 for INR 12,000.00.",
           "Invoice ZP-9 was already paid on 2026-03-01 under payment [PAY-9] matching ledger [LED-9] and remittance "
           "[REM-9].")
g = decide.finalize(PKG3, {"action": "reject_duplicate", "decisiveRefs": ["PAY-9", "LED-9", "REM-9"]})
check("real duplicate untouched", g["action"] == "reject_duplicate" and g["source"] == "model", g)

# structured facts come from the claimed invoice, not from prior-payment / archive sub-records
nested = {"packageId": "F", "priorPayments": [{"invoiceNumber": "OLD-1", "amountMinor": 5}],
          "claim": {"vendorName": "Acme", "invoiceNumber": "NEW-2", "amountMinor": 4520000, "currency": "INR"}}
check("facts skip prior payments", decide.build_facts(nested, {}) ==
      {"vendorName": "Acme", "invoiceNumber": "NEW-2", "amountMinor": 4520000, "currency": "INR"},
      decide.build_facts(nested, {}))
# a document titled "Archive" is noise even when its paragraph does not say so
titled = {"packageId": "T", "documents": [
    {"title": "Archive", "text": "Invoice T-1 was already paid [A-1] [A-2] [A-3]."},
    {"title": "Reconciliation", "text": "PO [P-1] and GRN [G-1] match invoice [I-1]; within authority, reconciled."}]}
g = decide.finalize(titled, {"action": "reject_duplicate", "decisiveRefs": ["A-1", "A-2", "A-3"], "invoiceNumber": "T-1"})
check("title marks archive as noise", g["action"] == "settle_invoice" and g["evidenceRefs"] == ["P-1", "G-1", "I-1"], g)
check("'Nothing is pending' is negated", decide.risk_signal(
    {"documents": [{"text": "All matched. Nothing is pending."}]}) is None)

# error classification: Groq TPM 429 waits only the advertised time; nothing disables a model for a day
class R:
    def __init__(self, code, text, headers=None):
        self.status_code, self.text, self.headers = code, text, headers or {}


check("429 uses 'try again in'", decide.classify(R(429, "Please try again in 10.97s."))[1] == 10.97)
check("413 -> split", decide.classify(R(413, "Request too large"))[0] == "too_large")
check("groq json failure -> retry", decide.classify(R(400, '{"error":{"code":"json_validate_failed"}}')) == ("bad_json", 0))
check("402 in-flight -> short wait", decide.classify(R(402, "This request would exceed your available credits given "
                                                          "your current in-flight requests."))[0] == "credit_busy"
      and decide.classify(R(402, "in-flight"))[1] <= 5)
check("402 no credit -> skip paid model", decide.classify(R(402, "Insufficient credits"))[0] == "no_credit")
check("gemini daily quota -> stop for an hour", decide.classify(R(429, '{"error":{"details":[{"violations":[{'
      '"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}'))[0] == "daily_quota")
check("gemini per-minute 429 waits retryDelay", decide.classify(R(429, "Please retry in 34.5s."))[1] == 34.5)
check("404 skip <= 5 min", decide.classify(R(404, "model not found"))[1] <= 300)

print("\nALL PASS" if not failures else f"\n{len(failures)} FAILURES: {failures}")
