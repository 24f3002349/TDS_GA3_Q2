"""Generated variants of the record-style audit layout (one reference per record, decision combines records).
Covers all five actions and their traps. Usage: python eval_records.py [-v]"""
import os
import random
import sys

if __name__ == "__main__":
    os.environ["LLM_API_KEY"] = ""
import decide  # noqa: E402

DECOYS = [
    "Archive examples mention every available action but govern no live case.",
    "Historical bank transfer for a different order contains the word duplicate, but no current-order identifier.",
    "A closed reconciliation incident for a separate supplier recorded a pricing dispute and subsequent correction.",
    "Archived supplier memo for unrelated order PO-7142; it does not refer to the current order.",
    "Operations discussed a bank-detail change in a training case, not this vendor's active instruction.",
    "A previous quarter used a different authorization threshold; that archived delegation has expired.",
]
CONTROLS = ("Payment controls: do not pay a commercial order already discharged; a material order/invoice difference "
            "requires the exception process; an unverified bank change pauses payment. Otherwise follow the delegation limit.")
FILLER = "The month-end report groups documents by arrival time rather than by the date printed on the invoice. Review location Pune; docket 1234."


def case(n, want, *, paid_order, order, signed, total, master, remit, callback, ceiling, wording=0, dispute=False):
    sup, cur = f"Vendor {n} Supplies", "USD"
    ledger = (f"Payment ledger: a disbursement for {cur} {signed:.2f} to {sup} previously settled order {paid_order}. "
              f"The currently submitted order is {order}.") if wording == 0 else (
              f"Ledger extract: {cur} {signed:.2f} was paid to {sup}, which settled order {paid_order}. "
              f"The currently submitted order is {order}.")
    order_rec = (f"Order {order} is signed for {cur} {signed:.2f}. The receiving entry confirms the billed units; "
                 f"submitted invoice total is {cur} {total:.2f}. No corrected invoice exists.")
    if dispute:
        order_rec += " The supplier has not resolved the valuation dispute."
    cb = ("The callback for this request is confirmed through the established vendor channel." if callback else
          "The callback to the established vendor channel is still pending.")
    master_rec = (f"Vendor master lists account {master}. Remittance instructions now specify {remit}. {cb} "
                  f"Approved autonomous payment ceiling for {cur} is {cur} {ceiling:.2f}. A named approver is required "
                  f"when the payable total is greater. {CONTROLS}")
    live = [ledger, order_rec, master_rec]
    refs = [f"R_L{n:02d}{i}" for i in range(3)]
    rng = random.Random(n)
    decoys = rng.sample(DECOYS, 5)
    docs = {"ledger-and-correspondence.txt": [], "policy-and-audit-notes.txt": [], "intake-and-cover-sheet.txt": []}
    names = list(docs)
    for i, (r, t) in enumerate(zip(refs, live)):
        docs[names[i]].append(f"[{r}] {t}")
    for j, t in enumerate(decoys):
        docs[names[j % 3]].insert(rng.randint(0, len(docs[names[j % 3]])), f"[R_D{n:02d}{j}] {t}")
    pkg = {"packageId": f"pkg_R{n:02d}", "documents": [
        {"name": k, "text": "\n\n".join(v + [FILLER])} for k, v in docs.items()]}
    return pkg, want, refs


C = []
add = lambda *a, **k: C.append(case(len(C) + 1, *a, **k))
add("settle_invoice", paid_order="PO-1", order="PO-2", signed=1000, total=1000, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("settle_invoice", paid_order="PO-11", order="PO-12", signed=1999.99, total=1999.99, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("settle_invoice", paid_order="PO-21", order="PO-22", signed=500, total=500, master="AC-1", remit="AC-9", callback=True, ceiling=900)
add("settle_invoice", paid_order="PO-31", order="PO-32", signed=750, total=750, master="AC-4", remit="AC-4", callback=True, ceiling=750, wording=1)
add("reject_duplicate", paid_order="PO-41", order="PO-41", signed=1000, total=1000, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("reject_duplicate", paid_order="PO-51", order="PO-51", signed=5000, total=5000, master="AC-1", remit="AC-1", callback=True, ceiling=2000, wording=1)
add("open_exception", paid_order="PO-61", order="PO-62", signed=1000, total=1100, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("open_exception", paid_order="PO-71", order="PO-72", signed=1000, total=1000, master="AC-1", remit="AC-1", callback=True, ceiling=2000, dispute=True)
add("open_exception", paid_order="PO-81", order="PO-82", signed=9000, total=8000, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("hold_invoice", paid_order="PO-91", order="PO-92", signed=1000, total=1000, master="AC-1", remit="AC-2", callback=False, ceiling=2000)
add("hold_invoice", paid_order="PO-93", order="PO-94", signed=3000, total=3000, master="AC-3", remit="AC-7", callback=False, ceiling=2000)
add("request_approval", paid_order="PO-95", order="PO-96", signed=2500, total=2500, master="AC-1", remit="AC-1", callback=True, ceiling=2000)
add("request_approval", paid_order="PO-97", order="PO-98", signed=2000.01, total=2000.01, master="AC-1", remit="AC-1", callback=True, ceiling=2000, wording=1)


def main():
    bad = 0
    for pkg, want, refs in C:
        r = decide.finalize(pkg, {"source": "fallback", **decide.heuristic_facts(pkg), **decide.heuristic(pkg)})
        ok = r["action"] == want and r["evidenceRefs"] == refs
        bad += not ok
        if not ok or "-v" in sys.argv:
            print(f"{'OK ' if ok else 'BAD'} {pkg['packageId']} want {want:17s} got {r['action']:17s} [{r['source']}] "
                  f"refs={r['evidenceRefs']} facts={r['facts']}")
    print(f"\n{len(C) - bad}/{len(C)} correct")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
