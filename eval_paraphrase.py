"""Generalisation check for the case reader: decisive paragraphs paraphrased by hand (not used to write the cues),
placed in the real case-file layout with archive/training decoys and filler. Usage: python eval_paraphrase.py"""
import os
import sys

if __name__ == "__main__":
    os.environ["LLM_API_KEY"] = ""
import decide  # noqa: E402

FILLER = ("The intake export preserves punctuation from the source system, so capitalization is not evidence of "
          "identity. The referenced office was Pune and the audit export sequence was 1234.\n\n"
          "A training example in the archive uses the phrase 'hold invoice'; it is not an instruction for this case. "
          "The referenced office was Kochi and the audit export sequence was 5678.")
CASES = [
    # settle_invoice
    ("settle_invoice", "Price and quantity on the invoice match the order and the goods receipt [A]. No payment for this "
     "invoice number appears in the ledger [B]. The claim is clear to pay within standing authority [C]."),
    ("settle_invoice", "A remittance to the same supplier exists, but it paid a different invoice with its own order [A]. "
     "The present invoice is supported by its own receipt [B]. Nothing has been paid against it, so it can be settled [C]."),
    ("settle_invoice", "Three-way matching succeeded with no variance [A]. The total is inside the operator's delegated "
     "limit [B]. Duplicate screening found no earlier payment for this invoice [C]."),
    ("settle_invoice", "The bank-detail confirmation call was completed and matched the registered number [A]. Order and "
     "receipt agree with the invoice [B]. No prior disbursement exists for it, and it may be paid now [C]."),
    ("settle_invoice", "Every billed line was received and accepted [A]. The ledger shows this invoice as open and unpaid "
     "[B]. Policy allows autonomous release for reconciled invoices under the ceiling [C]."),
    # request_approval
    ("request_approval", "All records reconcile cleanly [A]. The invoice total is higher than the autonomous payment "
     "threshold for this queue [B]. A designated approver must sign off before release [C]."),
    ("request_approval", "Order, receipt and invoice agree [A]. The amount sits above the agent's delegated authority [B]. "
     "Route the claim to the approval queue rather than paying it directly [C]."),
    ("request_approval", "Nothing is wrong commercially and no duplicate exists [A]. However the operator may only release "
     "payments up to USD 4100.50, below this total [B]. Larger amounts need a manager's approval [C]."),
    ("request_approval", "The claim is valid and fully matched [A]. Its value exceeds the self-service limit [B]. Escalate "
     "to a finance approver [C]."),
    ("request_approval", "Reconciliation is complete with no open issue [A]. This payment is beyond what the queue can "
     "authorise on its own [B]. Approval by the delegated signatory is required [C]."),
    # hold_invoice
    ("hold_invoice", "The supplier emailed new bank details last week [A]. Confirmation via the phone number on file is "
     "still outstanding [B]. Do not release funds until that verification is done [C]."),
    ("hold_invoice", "A request to redirect payment to another account arrived after the order [A]. The out-of-band "
     "confirmation has not come back yet [B]. Payment is on hold pending that check [C]."),
    ("hold_invoice", "Remittance details on the invoice differ from the vendor master [A]. Treasury is awaiting the "
     "callback to the known contact [B]. Release stays suspended until it succeeds [C]."),
    ("hold_invoice", "Tax registration of the supplier is still being verified with the authority [A]. The verification "
     "is not yet complete [B]. Policy holds payment until verification finishes [C]."),
    ("hold_invoice", "The vendor asked to change the beneficiary account [A]. Nobody has yet reached the vendor on the "
     "previously registered line [B]. Keep the payment paused until the identity check closes [C]."),
    # reject_duplicate
    ("reject_duplicate", "This invoice was already paid last month in full [A]. The new submission is a rescanned copy of "
     "the same invoice [B]. Paying again would be a duplicate payment, so reject it [C]."),
    ("reject_duplicate", "The ledger shows a settled payment with the same supplier, invoice number, currency and amount "
     "[A]. Only spacing in the identifier differs [B]. Policy forbids paying the same invoice twice [C]."),
    ("reject_duplicate", "A matching invoice was disbursed on 2026-03-02 [A]. The current file is a resubmission of that "
     "instrument, not a new invoice [B]. Duplicate control requires rejection [C]."),
    ("reject_duplicate", "Payment history already contains this exact invoice [A]. The intake is a second copy of the "
     "same document [B]. It must not be paid a second time [C]."),
    ("reject_duplicate", "The earlier settlement covered precisely this commercial invoice [A]. OCR noise is the only "
     "difference between the two images [B]. Reject the repeat claim as a duplicate [C]."),
    # open_exception
    ("open_exception", "The invoice charges a unit price above the contract rate [A]. The supplier and buyer disagree "
     "about which rate applies and no credit note exists [B]. Unresolved pricing disputes go to the exception queue [C]."),
    ("open_exception", "Quantity received is lower than the quantity billed [A]. The records contradict each other and "
     "no corrected invoice has been issued [B]. Open an exception case [C]."),
    ("open_exception", "The order currency and the invoice currency differ and the totals do not agree [A]. Neither party "
     "has explained the gap [B]. Material record conflicts require exception handling [C]."),
    ("open_exception", "The invoice total is outside the agreed variance tolerance against the receipt valuation [A]. "
     "Explanations from both sides are inconsistent [B]. Policy forbids paying or simply holding such a case [C]."),
    ("open_exception", "The goods receipt and the invoice describe different items [A]. The mismatch is material [B]. "
     "Escalate to the exception workflow, not to approval [C]."),
]


def package(n, decisive):
    refs = [f"R_P{n:02d}{x}" for x in "ABC"]
    for x, r in zip("ABC", refs):
        decisive = decisive.replace(f"[{x}]", f"[{r}]")
    docs = [
        {"name": "intake-and-cover-sheet.txt",
         "text": f"Case-file extract. Supplier Test Vendor {n}; invoice INV-2026-{n:06d}; stated total USD 1234.56. The "
                 f"normalized supplier and invoice identifiers were read from the signed cover sheet [R_CV{n:02d}].\n\n" + FILLER},
        {"name": "ledger-and-correspondence.txt", "text": decisive + "\n\n" + FILLER},
        {"name": "policy-and-audit-notes.txt",
         "text": f"Archive note [R_AR{n:02d}]: a closed, unrelated case mentioned 'settle immediately'; the archive marks "
                 f"it non-operative. Training appendix [R_TR{n:02d}]: examples list all five action names, but the "
                 f"appendix says they are vocabulary, not case instructions.\n\n" + FILLER},
    ]
    return {"packageId": f"pkg_P{n:02d}", "documents": docs}, refs



def main():
    bad = unsafe = unsure = 0
    for n, (want, text) in enumerate(CASES, 1):
        pkg, refs = package(n, text)
        case = decide.read_case(pkg)
        r = decide.finalize(pkg, {"source": "fallback", **decide.heuristic_facts(pkg), **decide.heuristic(pkg)})
        ok = r["action"] == want and r["evidenceRefs"] == refs and r["facts"]["vendorName"] == f"Test Vendor {n}"
        bad += not ok
        unsure += not case["confident"]
        unsafe += r["action"] == "settle_invoice" and want != "settle_invoice"
        if not ok or "-v" in sys.argv:
            print(f"{'OK ' if ok else 'BAD'} P{n:02d} want {want:17s} got {r['action']:17s} [{r['source']}] "
                  f"{decide.cue_scores(case['text'])}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} correct; not confident: {unsure}; unsafe settles: {unsafe}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
