"""Decision-quality eval on synthetic packages built like the grader's (decoys, negation, other-invoice payments,
superseded limits, resolved checks). Each case has a known action, decisive refs and facts.

Usage:  python eval_decide.py            (offline: rule-based fallback + guard only, no model)
        python eval_decide.py --model    (uses LLM_* settings from .env / environment)
Exit code 0 only if every case is right."""
import asyncio
import os
import sys

if "--model" in sys.argv and os.path.exists(".env"):
    for line in open(".env", encoding="utf-8"):
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())
if "--model" not in sys.argv:
    os.environ["LLM_API_KEY"] = ""

import decide  # noqa: E402

BOILER = ("Standard terms: payment within 30 days of a valid invoice; disputes must be raised in writing; this "
          "transmittal does not itself authorise any payment. Reviewers should read every attached record. ")
POLICY = "AP-2026.10"
LIMIT = "INR 5,00,000.00"


def case(n, want, vendor, inv, amount_text, decisive, decisive_refs, extra=(), currency="INR"):
    t = f"C{n:02d}"
    docs = [
        {"title": "Cover sheet", "text": f"Claim pack for {vendor}, invoice {inv}. Batch reference [CS-{t}]. {BOILER}"},
        {"title": "Invoice", "text": f"Tax invoice from {vendor}. Invoice number: {inv}. Invoice total: {currency} "
                                     f"{amount_text}. Due in 30 days."},
        {"title": "Archive",
         "text": f"ARCHIVE - 2023 worked example used in analyst training only: an invoice from another vendor was "
                 f"rejected as a duplicate after its remittance matched [ARC-{t}-1] [ARC-{t}-2] [ARC-{t}-3]. "
                 f"This example does not concern the current claim."},
        {"title": "Training note",
         "text": f"Training decoy: when a bank call-back is pending, analysts must hold payment; when totals exceed "
                 f"authority, request approval [TRN-{t}-1] [TRN-{t}-2] [TRN-{t}-3]. Not a record of this invoice."},
        {"title": "Authority policy",
         "text": f"Policy {POLICY} (current): the agent may settle autonomously up to {LIMIT} per invoice; above that a "
                 f"finance controller must approve. Superseded policy AP-2025.2 allowed INR 10,00,000.00 and no "
                 f"longer applies."},
        *[{"title": "Note", "text": e} for e in extra],
        {"title": "Controlling record", "text": decisive},
    ]
    return {"want": want, "refs": decisive_refs, "facts": (vendor, inv, currency),
            "pkg": {"packageId": f"EVAL-{t}", "documents": docs}}


def settle_text(t, amt):
    return (f"Three-way match confirmed: purchase order [PO-{t}], goods receipt [GRN-{t}] and invoice record "
            f"[IR-{t}] agree on quantity and price for the INR {amt} total, which is within the current autonomous "
            f"authority. Nothing is pending.", [f"PO-{t}", f"GRN-{t}", f"IR-{t}"])


CASES = []


def add(want, vendor, inv, amt, kind, extra=(), currency="INR"):
    n = len(CASES) + 1
    t = f"C{n:02d}"
    if kind == "settle":
        text, refs = settle_text(t, amt)
    elif kind == "approval":
        text, refs = (f"PO [PO-{t}] and goods receipt [GRN-{t}] reconcile with the invoice, but its total of INR {amt} "
                      f"exceeds the autonomous authority under current policy [POL-{t}], so a finance controller must "
                      f"approve before payment.", [f"PO-{t}", f"GRN-{t}", f"POL-{t}"])
    elif kind == "hold":
        text, refs = (f"The vendor changed its bank account; under control [CTL-{t}] payment of invoice record "
                      f"[IR-{t}] is paused until the call-back verification [VER-{t}] to the registered phone number "
                      f"is completed. The call-back has not happened yet.", [f"CTL-{t}", f"IR-{t}", f"VER-{t}"])
    elif kind == "hold_tax":
        text, refs = (f"GSTIN validation for this vendor is still awaiting the tax portal response [GST-{t}]; per "
                      f"control [CTL-{t}] invoice record [IR-{t}] must not be paid until that verification completes.",
                      [f"GST-{t}", f"CTL-{t}", f"IR-{t}"])
    elif kind == "dup":
        text, refs = (f"Remittance advice [REM-{t}] shows invoice {inv} from {vendor} for {currency} {amt} was already paid on "
                      f"2026-08-14 in payment run [PAY-{t}]; the bank ledger [LED-{t}] confirms the debit.",
                      [f"REM-{t}", f"PAY-{t}", f"LED-{t}"])
    elif kind == "exception_qty":
        text, refs = (f"Goods receipt [GRN-{t}] records 40 units received, but invoice record [IR-{t}] bills 55 units "
                      f"against purchase order [PO-{t}]; the quantity conflict is material and needs an exception "
                      f"review.", [f"GRN-{t}", f"IR-{t}", f"PO-{t}"])
    elif kind == "exception_price":
        text, refs = (f"Purchase order [PO-{t}] fixes the unit price at INR 1,200.00 but invoice record [IR-{t}] charges "
                      f"INR 1,450.00 per unit; the contract register [CON-{t}] shows no approved price change.",
                      [f"PO-{t}", f"IR-{t}", f"CON-{t}"])
    else:
        raise ValueError(kind)
    CASES.append(case(n, want, vendor, inv, amt, text, refs, extra, currency))


# --- settle, with traps that tempt other actions -------------------------------------------------------------
add("settle_invoice", "Acme Tools Pvt Ltd", "INV-7781", "45,200.00", "settle")
add("settle_invoice", "Blue Fern Traders", "BFT-0912", "1,18,000.00", "settle",
    ["Payment [PAY-OLD] of INR 1,18,000.00 on 2026-07-02 settled invoice BFT-0911, the previous month's invoice; it "
     "is a different invoice."])
add("settle_invoice", "Kaveri Office Supply", "KOS-332", "23,450.50", "settle",
    ["Duplicate check: no duplicate payment was found for invoice KOS-332 in the last 24 months."])
add("settle_invoice", "Northwind Metals", "NWM-5521", "3,75,000.00", "settle",
    ["The bank-detail call-back [VER-OK] was completed and confirmed on 2026-09-30; no verification is pending."])
add("settle_invoice", "Hold Fast Fasteners Ltd", "HFF-77", "4,99,999.00", "settle",
    ["Vendor product line: 'Hold Fast' anchors and 'Approval Grade' bolts, as listed in the catalogue."])
add("settle_invoice", "Orion Logistics", "ORL-4410", "2,10,000.00", "settle",
    ["Earlier quantity mismatch on this PO was resolved: the corrected goods receipt was posted and the mismatch "
     "cleared on 2026-09-12."])
# --- request_approval --------------------------------------------------------------------------------------
add("request_approval", "Blue River Logistics", "BRL-220", "9,80,000.00", "approval")
add("request_approval", "Sapphire Infra", "SPI-0090", "6,25,000.00", "approval",
    ["Old note: under superseded policy AP-2025.2 this amount could have been settled automatically."])
add("request_approval", "Delta Foods", "DF-4471", "5,00,000.01", "approval")
# --- hold_invoice ----------------------------------------------------------------------------------------------
add("hold_invoice", "Zenith Paper Co", "ZPC-19", "48,000.00", "hold")
add("hold_invoice", "Lotus Chemicals", "LC-7001", "2,40,000.00", "hold_tax",
    ["PO and goods receipt match the invoice, and the amount is within autonomous authority."])
add("hold_invoice", "Ganga Textiles", "GT-118", "12,500.00", "hold",
    ["No duplicate payment exists for GT-118; payment [PAY-GT] was for invoice GT-117."])
# --- reject_duplicate ------------------------------------------------------------------------------------------
add("reject_duplicate", "Zen Paper Co", "ZP-9", "12,000.00", "dup")
add("reject_duplicate", "Indus Electricals", "IE-2208", "86,400.00", "dup",
    ["PO and goods receipt fully reconcile with this invoice and the amount is within autonomous authority."])
add("reject_duplicate", "Meridian Labs", "ML-55", "1,500.00", "dup",
    ["Second copy of the invoice was re-sent by the vendor by email on 2026-09-01 asking for payment."], "USD")
# --- open_exception --------------------------------------------------------------------------------------------
add("open_exception", "Coastal Cement", "CC-9002", "3,30,000.00", "exception_qty")
add("open_exception", "Pioneer Pumps", "PP-713", "87,000.00", "exception_price",
    ["Approval note: the amount is within autonomous authority, so no approver is needed."])
add("open_exception", "Vega Plastics", "VP-61", "54,000.00", "exception_qty",
    ["A bank call-back for this vendor was completed last quarter; nothing is pending on bank details."])

MINOR = {"USD": 100, "INR": 100}


async def main():
    class NoStore:
        def get_decision(self, k):
            return None

        def put_decision(self, k, v):
            pass

    pkgs = [c["pkg"] for c in CASES]
    loop = asyncio.get_running_loop()
    results = await decide.decide_packages(pkgs, loop.time() + 120, NoStore(),
                                           {"policyRevision": POLICY})
    bad, unsafe = 0, 0
    for c, r in zip(CASES, results):
        vendor, inv, cur = c["facts"]
        amt = c["pkg"]["documents"][1]["text"].split(cur + " ")[1].split(".")[0] + "." + \
            c["pkg"]["documents"][1]["text"].split(cur + " ")[1].split(".")[1][:2]
        want_minor = decide.to_minor(amt, cur)
        f = r["facts"]
        problems = []
        if r["action"] != c["want"]:
            problems.append(f"action {r['action']} != {c['want']}")
            if r["action"] == "settle_invoice":
                unsafe += 1
        refs = [x.strip("[]") for x in r["evidenceRefs"]]
        if refs != c["refs"]:
            problems.append(f"refs {refs} != {c['refs']}")
        if (f["vendorName"].rstrip("."), f["invoiceNumber"], f["amountMinor"], f["currency"]) != \
                (vendor, inv, want_minor, cur):
            problems.append(f"facts {f} != {(vendor, inv, want_minor, cur)}")
        if not (60 <= len(r["rationale"]) <= 1500) or sum(f"[{x}]" in r["rationale"] for x in refs) < 2:
            problems.append("rationale")
        bad += bool(problems)
        print(("OK  " if not problems else "BAD ") + f"{c['pkg']['packageId']} {c['want']:17s} [{r['source']}] "
              + "; ".join(problems))
    print(f"\n{len(CASES) - bad}/{len(CASES)} fully correct; unsafe settles: {unsafe}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    asyncio.run(main())
