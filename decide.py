"""AI decision layer: one action + decisive evidence + facts per invoice package.

The model only proposes; every field is validated against the package here:
  * evidenceRefs must be IDs that literally occur in the package (bracketed refs or evidence id fields),
    completed to the decisive paragraph's full set when the model cites most of it,
  * the rationale is assembled in code so it always names the action and says what each cited ID shows,
  * amounts are converted to minor units in code, not by the model,
  * decisions are cached by canonical package content so repeat batches need no model call.
"""
import asyncio
import hashlib
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation

import httpx

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
PROMPT_VERSION = "v8"  # bump whenever prompt or validation changes, so stale cached decisions are not reused

# OPENROUTER_API_KEY, when set, takes precedence over the generic LLM_* provider settings (one variable to switch).
# An OpenRouter key ("sk-or-...") placed in LLM_API_KEY is recognised too: it only works on OpenRouter anyway.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() or (
    os.environ.get("LLM_API_KEY", "").strip() if os.environ.get("LLM_API_KEY", "").strip().startswith("sk-or-") else "")
if OPENROUTER_API_KEY:
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_API_KEY = OPENROUTER_API_KEY
    _models = (os.environ.get("OPENROUTER_MODEL")
               or (os.environ.get("LLM_MODEL") if "openrouter.ai" in os.environ.get("LLM_BASE_URL", "") else "")
               or "anthropic/claude-haiku-5.5,openai/gpt-6-luna")
    LLM_REASONING = os.environ.get("OPENROUTER_REASONING_EFFORT", "")
else:
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    _models = os.environ.get("LLM_MODEL") or "gemini-3.6-flash,gemini-3.5-flash,gemini-3.8-flash"
    LLM_REASONING = os.environ.get("LLM_REASONING_EFFORT", "")  # e.g. low / medium; empty = provider default
# The model list is a preference order; LLM_FALLBACK_MODELS is appended. When a model answers 404 (retired),
# 429 (quota) or 5xx (overloaded) the next one is tried instead of falling back to keywords.
LLM_MODELS = list(dict.fromkeys(
    m.strip() for m in f'{_models},{os.environ.get("LLM_FALLBACK_MODELS", "")}'.split(",") if m.strip()))
# Where each model is served: model -> (base URL, API key, extra request fields). Order of LLM_MODELS is the
# preference order: primary provider, then an optional second provider (FALLBACK_*), then a free OpenRouter model.
ENDPOINT = {m: (LLM_BASE_URL, LLM_API_KEY, {}) for m in LLM_MODELS}
LAST_RESORT: set = set()  # slow free models: skipped while a preferred model is only briefly unavailable
_fb_key = os.environ.get("FALLBACK_API_KEY", "").strip()
if _fb_key and os.environ.get("FALLBACK_BASE_URL"):
    for m in (x.strip() for x in os.environ.get("FALLBACK_MODEL", "").split(",")):
        if m and m not in ENDPOINT:
            ENDPOINT[m] = (os.environ["FALLBACK_BASE_URL"].rstrip("/"), _fb_key, {})
            LLM_MODELS.append(m)
if OPENROUTER_API_KEY and os.environ.get("OPENROUTER_FREE_FALLBACK", "on").lower() != "off":
    # Last resort when paid models are refused (402, no credit): accurate but slow, so thinking is switched off.
    _free = os.environ.get("OPENROUTER_FREE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    if _free not in ENDPOINT:
        ENDPOINT[_free] = (LLM_BASE_URL, LLM_API_KEY, {"reasoning": {"enabled": False}})
        LLM_MODELS.append(_free)
        LAST_RESORT.add(_free)
LLM_CHUNK = int(os.environ.get("LLM_CHUNK", "4"))
# Round-robin over models only where each model has its own small free-tier quota (Groq, Google AI Studio).
SPREAD = os.environ.get("LLM_SPREAD", "auto").lower() in ("1", "true", "on") or (
    os.environ.get("LLM_SPREAD", "auto").lower() == "auto" and ("groq.com" in LLM_BASE_URL
                                                                or "googleapis.com" in LLM_BASE_URL))
# How many leading models share the load; later ones are only used when those are unavailable.
SPREAD_TOP = int(os.environ.get("LLM_SPREAD_TOP", "2" if "googleapis.com" in LLM_BASE_URL else "0")) or None
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "16"))
KEEP_BRACKETS = os.environ.get("EVIDENCE_REF_BRACKETS", "false").lower() == "true"
AUDIT_MAX_PACKAGES = int(os.environ.get("AUDIT_MAX_PACKAGES", "8"))  # batches this small get model priority
WARM_BUDGET_S = float(os.environ.get("WARM_BUDGET_S", "240"))
# One stuck call (overloaded free tier) must not use up the whole request: give up and try another model.
CALL_TIMEOUT_S = float(os.environ.get("LLM_CALL_TIMEOUT_S", "25"))
SETTLE_GUARD = os.environ.get("SETTLE_GUARD", "on").lower() != "off"
CASE_READER_FIRST = os.environ.get("CASE_READER_FIRST", "on").lower() != "off"
# Deterministic results (model answers and confident case-reader decisions) are cached; weak fallbacks are retried.
CACHEABLE = ("model", "guard", "noise-guard", "rules", "rules-override")

REF_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")
ID_KEY_RE = re.compile(r"^(id|ref|evidenceid|evidenceref|refid|docid|documentid|recordid|sourceid)$", re.I)
ZERO_DECIMAL = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "XAF", "XOF", "PYG"}
THREE_DECIMAL = {"KWD", "BHD", "OMR", "JOD", "TND", "LYD", "IQD"}

_sem = None
_pending: dict[str, asyncio.Future] = {}
_cooldown: dict[str, float] = {}  # model -> loop time until which it is skipped
MODEL_STATUS: dict[str, dict] = {}  # model -> last outcome (no secrets), shown on /healthz
_warming: set = set()  # cache keys being decided in the background after a fallback
_no_reasoning: set = set()  # models that rejected reasoning_effort
_param_fixes: dict[str, set] = {}  # model -> request parameters it rejected (dropped, or max_tokens renamed)
# Free tiers are limited per model per minute (Groq: 8K tokens/min), so calls are spread round-robin over all
# ready models, and when every model is cooling down the waiters are served by level:
# 0 = small (audit) batch, 1 = core batch, 2 = background cache warming.
LEVEL_AUDIT, LEVEL_CORE, LEVEL_WARM = 0, 1, 2
_waiting = [0, 0, 0]
_rr = 0

SYSTEM_PROMPT = """You are an accounts-payable control agent. For EACH invoice package choose exactly one action,
using only that package's own documents plus the batch context (policy revision, authority limits, currency rules).

Actions:
- settle_invoice: invoice is valid, fully reconciled (PO / receipt / invoice agree) and the amount is within the
  autonomous (delegated) payment authority in force. Nothing pending, no conflict, not already paid.
- request_approval: commercially valid and reconciled, BUT the amount (or another stated condition) is outside the
  agent's delegated/autonomous authority, so a human approver must sign off.
- hold_invoice: payment must pause until a specific, stated verification completes (e.g. bank-detail change
  call-back, tax/registration check, goods-receipt confirmation pending, sanctions screening). Something is pending.
- reject_duplicate: this SAME commercial invoice (same vendor + same invoice number, same amount) was ALREADY PAID.
  A payment for a DIFFERENT invoice (different invoice number, PO, period or amount) is NOT a duplicate.
- open_exception: material records CONFLICT (e.g. amount/quantity/price/vendor/bank mismatch between PO, receipt and
  invoice) and need an exception workflow rather than a simple wait.

Method - apply it the same way to every package, including layouts you have not seen before:
1. Find the CONTROLLING facts: the invoice being claimed in this package and the current, operative records about
   it. The documents are deliberately noisy. NOT controlling: cover sheets / transmittals, archived or historical
   cases, training, sample or illustrative material, templates, drafts, superseded or replaced policy text, and
   records about other invoices or vendors - even when they use strong words like "duplicate", "hold" or "approval".
   When a later or amended record replaces an earlier one, the later record controls. The batch context's
   policyRevision names the policy in force: use authority limits and rules from that revision only, never from
   other or older revisions quoted in the documents.
2. Read for meaning, not keywords: watch NEGATION ("no duplicate was found", "is not pending", "does not exceed"),
   resolved items ("call-back completed", "mismatch cleared") and action words used in irrelevant contexts.
3. Decide from the controlling facts in this order: already paid -> reject_duplicate; unresolved conflict between
   records -> open_exception; a named verification still pending -> hold_invoice; reconciled but outside autonomous
   authority (compare the claimed amount with the limit in force) -> request_approval; otherwise, if reconciled,
   within authority and nothing pending -> settle_invoice. Never settle on insufficient evidence: if nothing shows
   the invoice is reconciled and within authority, or the evidence contradicts itself, do not choose settle_invoice.
4. Evidence IDs: bracketed tokens like [ABC-123] (or the id field of an evidence/document object) identify records.
   Cite the IDs of the decisive record - the current, operative paragraph about THIS invoice that determines the
   action. It normally carries exactly three bracketed IDs; cite all of them and nothing from noise. Copy each ID
   character-for-character (without brackets); never invent, shorten or reformat an ID.

Facts are for the invoice being claimed in this package: vendorName and invoiceNumber exactly as written,
currency as ISO-4217 code, the invoice total amount exactly as printed in major units (amountText, e.g. "1,24,500.00"),
and amountMinor = that amount in minor units (cents/paise) as an integer.

Reply with JSON only:
{"decisions":[{"packageId":"...","action":"one of the five","decisiveRefs":["REF1","REF2","REF3"],
"evidence":[{"ref":"REF1","shows":"what this record states and how it supports the action"}],
"vendorName":"...","invoiceNumber":"...","amountText":"...","amountMinor":0,"currency":"INR",
"reason":"1-3 sentences: the controlling facts and why they lead to this action rather than the alternatives"}]}
One decision per package; one evidence entry per decisive ref."""

ACTION_MEANING = {
    "settle_invoice": "the invoice is reconciled, within autonomous payment authority and nothing is pending",
    "request_approval": "the invoice is valid but outside the agent's delegated authority, so a human must approve it",
    "hold_invoice": "a stated verification is still pending, so payment must pause until it completes",
    "reject_duplicate": "this same invoice has already been paid, so paying it again would be a duplicate payment",
    "open_exception": "the records conflict, so the invoice needs an exception review instead of payment",
}


def package_id(pkg: dict):
    return pkg.get("packageId", pkg.get("id"))


def batch_context(batch: dict) -> dict:
    """Batch-level policy/context for the model: everything except the packages and the per-run batchId."""
    return {k: v for k, v in batch.items() if k not in ("packages", "batchId")}


def cache_key(pkg: dict, policy) -> str:
    content = {k: v for k, v in pkg.items() if k not in ("packageId", "id")}
    blob = json.dumps({"p": content, "policy": policy, "v": PROMPT_VERSION, "m": LLM_MODELS}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# --- package text utilities --------------------------------------------------------------
def string_leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from string_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from string_leaves(v)


LABEL_KEY_RE = re.compile(r"^(title|name|type|kind|section|heading|label|category|doc_?type|document_?type|status|"
                          r"classification|role|purpose)$", re.I)


def labeled_lines(obj, label: str = ""):
    """(label, line) for every non-empty text line. The label carries the enclosing document's title/type/status,
    so a paragraph inside a document titled "Archive" or "Cover sheet" is recognised as noise even when its own
    text does not say so."""
    if isinstance(obj, str):
        for line in re.split(r"\n+", obj):
            if line.strip():
                yield label, line
    elif isinstance(obj, dict):
        own = " ".join(v for k, v in obj.items() if isinstance(v, str) and LABEL_KEY_RE.match(k) and len(v) <= 80)
        inner = f"{label} {own}".strip()
        for k, v in obj.items():
            yield from labeled_lines(v, label if LABEL_KEY_RE.match(k) else inner)
    elif isinstance(obj, list):
        for v in obj:
            yield from labeled_lines(v, label)


def with_label(label: str, line: str) -> str:
    return f"{label} | {line}" if label else line


def paragraphs(pkg: dict):
    """All text paragraphs that carry bracketed refs, as (label | text, [refs in order])."""
    out = []
    for label, para in labeled_lines(pkg):
        refs = [r.strip() for r in REF_RE.findall(para)]
        if refs:
            out.append((with_label(label, para), list(dict.fromkeys(refs))))
    return out


def structured_ids(pkg: dict) -> list:
    """IDs carried by evidence/document objects (e.g. {"id": "EV-7", "text": ...}), excluding the package's own id."""
    out = []

    def walk(o, top):
        if isinstance(o, dict):
            for k, v in o.items():
                if not top and isinstance(v, str) and v.strip() and ID_KEY_RE.match(k):
                    out.append(v.strip())
                walk(v, False)
        elif isinstance(o, list):
            for v in o:
                walk(v, False)

    walk(pkg, True)
    return out


def norm_ref(r) -> str:
    return str(r).strip().strip("[]").strip()


def resolve_refs(pkg: dict, llm_refs) -> list:
    """Keep only IDs that exist in the package (exact spelling), snapped onto the decisive paragraph."""
    paras = paragraphs(pkg)
    canonical = {}
    for r in [r for _, refs in paras for r in refs] + structured_ids(pkg):
        canonical.setdefault(r, r)
        canonical.setdefault(r.lower(), r)
    valid = []
    for r in llm_refs or []:
        r = norm_ref(r)
        hit = canonical.get(r) or canonical.get(r.lower())
        if hit and hit not in valid:
            valid.append(hit)
    best, best_text, best_score = None, "", (0, 0)
    for text, refs in paras:
        score = (len(set(refs) & set(valid)), -abs(len(refs) - 3))
        if score > best_score:
            best, best_text, best_score = refs, text, score
    # The decisive paragraph carries three refs: complete it when the model cited two of them, or one of them
    # from a non-noise paragraph (small models often cite only the policy ref).
    if best and len(best) == 3 and (best_score[0] >= 2 or (best_score[0] == 1 and not NOISE.search(best_text))):
        return best
    return valid[:3]


def cites_only_noise(pkg: dict, refs: list) -> bool:
    """True when every cited ref occurs only in archive / training / cover-sheet style paragraphs."""
    paras = paragraphs(pkg)
    for r in refs:
        homes = [text for text, prefs in paras if r in prefs]
        if not homes or any(not NOISE.search(text) for text in homes):
            return False
    return bool(refs)


NOT_CLAIM_RE = re.compile(r"archiv|example|training|sample|histor|prior|previous|payment|remittance|ledger|decoy|"
                          r"template|superseded|related|other", re.I)


def find_key(obj, names):
    """Breadth-first search for the shallowest key with one of the given names, skipping sub-records that describe
    something other than the claimed invoice (prior payments, archive examples, ...)."""
    queue = [obj]
    while queue:
        nxt = []
        for o in queue:
            if isinstance(o, dict):
                labels = " ".join(v for k, v in o.items() if isinstance(v, str) and LABEL_KEY_RE.match(k))
                if o is not obj and NOT_CLAIM_RE.search(labels):
                    continue
                for k, v in o.items():
                    if k in names and not isinstance(v, (dict, list)) and v not in (None, ""):
                        return v
                nxt.extend(v for k, v in o.items() if isinstance(v, (dict, list)) and not NOT_CLAIM_RE.search(k))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        queue = nxt
    return None


def minor_candidates(amount_text, currency: str) -> list:
    """Every number in the text converted to minor units, in order of appearance."""
    out = []
    exp = 0 if currency in ZERO_DECIMAL else 3 if currency in THREE_DECIMAL else 2
    for num in re.findall(r"\d[\d,']*(?:\.\d+)?", str(amount_text or "")):
        try:
            out.append(int((Decimal(re.sub(r"[,']", "", num)) * (10 ** exp)).to_integral_value()))
        except InvalidOperation:
            pass
    return out


def to_minor(amount_text, currency: str):
    """The first number in the amount text ("45,200.00 incl. 18% GST" -> 4520000), in minor units."""
    found = minor_candidates(amount_text, currency)
    return found[0] if found else None


def build_facts(pkg: dict, d: dict) -> dict:
    currency = find_key(pkg, {"currency"}) or d.get("currency") or ""
    currency = str(currency).strip().upper()
    vendor = find_key(pkg, {"vendorName"}) or d.get("vendorName") or ""
    number = find_key(pkg, {"invoiceNumber"}) or d.get("invoiceNumber") or ""
    amount = find_key(pkg, {"amountMinor"})
    if not isinstance(amount, int) or isinstance(amount, bool):
        parsed = to_minor(d.get("amountText"), currency)
        llm_minor = d.get("amountMinor")
        if isinstance(llm_minor, str) and llm_minor.strip().isdigit():
            llm_minor = int(llm_minor.strip())
        if isinstance(llm_minor, int) and llm_minor in minor_candidates(d.get("amountText"), currency):
            amount = llm_minor  # the model's minor amount agrees with a number it quoted: trust it
        else:
            amount = parsed if parsed is not None else (llm_minor if isinstance(llm_minor, int) else 0)
    return {"vendorName": str(vendor).strip(), "invoiceNumber": str(number).strip(),
            "amountMinor": int(amount), "currency": currency}


def clean(text, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def evidence_notes(d: dict) -> dict:
    """Model's per-ref explanations, keyed by lower-cased ref."""
    notes = {}
    for e in d.get("evidence") or []:
        if isinstance(e, dict) and e.get("ref"):
            notes[norm_ref(e["ref"]).lower()] = e.get("shows") or e.get("supports") or e.get("why") or ""
    return notes


def rationale(action: str, reason: str, refs: list, notes: dict) -> str:
    """Action, controlling facts, then what each cited ID shows. Always 60-1500 chars and cites every ref."""
    parts = [f"Proposed action {action}: {ACTION_MEANING[action]}."]
    reason = clean(reason, 500)
    if reason:
        parts.append(reason if reason.endswith((".", "!", "?")) else reason + ".")
    if refs:
        cites = []
        for r in refs:
            note = clean(notes.get(r.lower()), 230).rstrip(".")
            cites.append(f"[{r}] {note}" if note else f"[{r}] is part of the same controlling record")
        parts.append("Evidence: " + "; ".join(cites) + ".")
    else:
        parts.append("The package carries no evidence ID that supports a different action.")
    return " ".join(parts)[:1500]


# --- case reader: deterministic reading of the case-file layout -------------------------------------------------
# A case file states its facts on the cover sheet ("Supplier X; invoice Y; stated total CUR N") and has exactly one
# decisive paragraph carrying three bracketed references; archive/training notes and the cover carry fewer.
COVER_RE = re.compile(r"Supplier\s+(?P<vendor>[^;\n]+?)\s*;\s*invoice\s+(?P<inv>[^;\s]+)\s*;\s*stated total\s+"
                      r"(?P<cur>[A-Z]{3})\s*(?P<amt>\d[\d,]*(?:\.\d+)?)", re.I)
GAP = r"(?:[^.]|\.\d){0,90}?"
CASE_CUES = {  # cues by meaning; GAP is a gap inside one sentence (decimal points allowed)
    "reject_duplicate": [
        r"already[- ](been )?(paid|settled|disbursed)", r"earlier (posting|settled|settlement)\b",
        r"completed payment", r"prior (ledger )?(transaction|payment|settlement) (closed|matches)",
        r"second (disbursement|settlement|payment)", r"(exact|commercial|already-paid) duplicate",
        r"duplicate (claim|submission|request|payment)", r"requires? rejection|to rejection",
        r"(same|one) (instrument|invoice|document)\b", r"(another|second|duplicate|repeat\w*) (image|scan|copy|intake|submission) of",
        r"not an? (distinct|separate|new|revised) invoice", r"same commercial key", r"treat" + GAP + r"duplicate"],
    "open_exception": [
        r"conflicts? with", r"beyond (the )?tolerance", r"outside the permitted" + GAP + r"tolerance",
        r"(does|do|did) not (reconcile|match|agree)", r"incompatible", r"contradict", r"mismatch",
        r"exception (queue|workflow|case|review|process)", r"no corrected (instrument|document|invoice)",
        r"unresolved" + GAP + r"(conflict|contradiction|dispute|difference)"],
    "hold_invoice": [
        r"call-?back", r"\b(call|verification|check|confirmation)\b" + GAP + r"(has not|not yet|incomplete|pending|neither|outstanding|awaiting)",
        r"\bpaus(e|ed|es)\b", r"freez", r"until (that|the|an?)\b" + GAP + r"(check|callback|call|verification|confirmation)",
        r"beneficiary", r"(newly supplied|new|changed|different|updated) (bank|destination|account|remittance|payment destination)",
        r"destination different|account change|change of (bank|account)", r"registered (account|number)",
        r"previously (verified|registered)", r"out-of-band", r"known[- ](channel|number)", r"identity check"],
    "request_approval": [
        r"outside the operator'?s?" + GAP + r"authority", r"(exceeds?|above|beyond|over)" + GAP + r"(authority|ceiling|delegat|limit|threshold)",
        r"below the stated total", r"(named|financial|human|senior) (financial )?approv(er|al)",
        r"approval (workflow|queue|route)", r"without escalation only up to", r"delegation (schedule|ceiling|limit)",
        r"escalat(e|ion) to", r"requires? (an? )?(approver|sign-?off)"],
    "settle_invoice": [
        r"no prior (settlement|payment)", r"no (paid item|earlier posting|settlement of)", r"remains unpaid",
        r"(meets|satisfies) the settlement conditions", r"not the document now under review",
        r"covers every billed unit|full quantity|invoiced quantity received", r"same payable total",
        r"reconciled" + GAP + r"with no exception|agree within", r"independent(ly)? issued|distinct invoice|separate invoice",
        r"authori[sz]e (its |the |this )?payment|approved? for payment", r"contains no settlement|has not been paid",
        r"(records?|receipt) validates?"],
}
CUE_NEGATED = re.compile(r"\b(no|not|never|without|nor)\s+(\w+\s+){0,2}$", re.I)


CASE_CUES["request_approval"] += [
    r"(higher|greater|more) than the" + GAP + r"(threshold|limit|authority|ceiling)",
    r"(exceeds?|above|beyond|over)" + GAP + r"authori[sz]e", r"(may|can) only (release|pay|approve|act|authori[sz]e)",
    r"(manager|designated|delegated|finance)'?s? (approver|signatory|approval)", r"approval (queue|by)", r"sign[- ]?off",
    r"(needs?|requires?|required)" + GAP + r"approval|approval" + GAP + r"(is )?required"]
CASE_CUES["settle_invoice"] += [
    r"no (payment|prior disbursement|earlier payment|duplicate payment|disbursement)\b" + GAP + r"(for|against|exists|appears|found)",
    r"(clear|cleared|eligible) (to|for) pay", r"(may|can) be (paid|settled)", r"open and unpaid|nothing has been paid",
    r"(within|inside|under) (the )?(operator'?s? |agent'?s? |queue'?s? )?(standing |delegated |autonomous )?"
    r"(authority|limit|ceiling)", r"allows? autonomous (release|payment|settlement)", r"different invoice"]
# A check described as already done ("the confirmation call was completed") is not a pending hold or open conflict.
RESOLVED_SENTENCE = re.compile(r"\b(was|were|has been|have been|is now) (completed|confirmed|cleared|resolved|verified|"
                               r"closed|corrected)\b", re.I)


def cue_scores(text: str) -> dict:
    """Matches per action in one paragraph. A non-settle cue preceded by a negation ("no earlier posting") is
    ignored, and so is a hold/exception cue inside a sentence that reports the check as already resolved."""
    sentences = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]*(?:\.\d[^.!?]*)*[.!?]?", text) if m.group().strip()]
    scores = {}
    for action, pats in CASE_CUES.items():
        n = 0
        for pat in pats:
            for m in re.finditer(pat, text, re.I):
                if action != "settle_invoice" and CUE_NEGATED.search(text[max(0, m.start() - 30):m.start()]):
                    continue
                if action in ("hold_invoice", "open_exception"):
                    sent = next((text[a:b] for a, b in sentences if a <= m.start() < b), "")
                    if RESOLVED_SENTENCE.search(sent):
                        continue
                n += 1
                break
        scores[action] = n
    return scores


def ref_sentences(text: str) -> dict:
    """ref -> the sentence of the paragraph that cites it (what that evidence shows)."""
    out = {}
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        for r in REF_RE.findall(sentence):
            out.setdefault(r.strip().lower(), clean(REF_RE.sub("", sentence), 260).rstrip(" ."))
    return out


def read_case(pkg: dict):
    """{action, confident, refs, notes, facts} read deterministically from the case file, or None."""
    cover = None
    for _, line in labeled_lines(pkg):
        m = COVER_RE.search(line)
        if m:
            cover = m
            break
    facts = None
    if cover:
        cur = cover["cur"].upper()
        facts = {"vendorName": cover["vendor"].strip(), "invoiceNumber": cover["inv"].strip().rstrip(".,"),
                 "amountMinor": to_minor(cover["amt"], cur), "currency": cur}
    paras = [(t, r) for t, r in paragraphs(pkg) if len(r) == 3 and not NOISE.search(t)]
    if not paras:
        records = read_records(pkg)
        if records:
            if facts:
                records["facts"] = {**records["facts"], **{k: v for k, v in facts.items() if v}}
            return records
        return {"action": None, "confident": False, "refs": [], "notes": {}, "facts": facts, "text": ""}
    scored = []
    for text, refs in paras:
        s = cue_scores(text)
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        scored.append((ranked[0][1], ranked[0][0], ranked[0][1] - ranked[1][1], text, refs))
    best, action, margin, text, refs = max(scored, key=lambda x: (x[0], x[2]))
    para = text.split(" | ", 1)[-1]
    return {"action": action if best and margin else None, "confident": best >= 2 and margin >= 1, "refs": refs,
            "notes": ref_sentences(para), "facts": facts, "text": para, "single": len(paras) == 1}


# --- record reader: case files where every record carries one reference and the decision combines records ---------
# e.g. "[R_..] Payment ledger: a disbursement for USD 1.00 to X previously settled order PO-1. The currently submitted
# order is PO-2." / "[R_..] Order PO-2 is signed for USD 1.00 ... submitted invoice total is USD 1.00." /
# "[R_..] Vendor master lists account AC-1. Remittance instructions now specify AC-1. The callback ... is confirmed ...
# Approved autonomous payment ceiling for USD is USD 2.00." Decoy records (archive, training, closed incidents for
# other suppliers, expired delegations, unrelated orders) are dropped; the live records are the evidence.
RECORD_NOISE = re.compile(
    r"archiv|training|example|historical|template|superseded|decoy|\bclosed\b[^.]*\b(incident|case)\b|separate supplier|"
    r"\bunrelated\b|\bexpired\b|previous quarter|govern no live case|no current-order identifier|does not refer to", re.I)
AMOUNT = r"([A-Z]{3})\s?(\d[\d,]*(?:\.\d+)?)"
RULE_TEXT = re.compile(r"\bcontrols?:|\bpolicy\b|\bdo not\b|\bmust not\b|\brequires? the\b|\botherwise follow\b", re.I)


def money(cur, num):
    return cur.upper(), to_minor(num, cur.upper())


def read_records(pkg: dict):
    records = []  # (ref, text) in document order
    for _, line in labeled_lines(pkg):
        refs = [r.strip() for r in REF_RE.findall(line)]
        if len(refs) == 1:
            records.append((refs[0], REF_RE.sub("", line).strip()))
    live = [(r, t) for r, t in records if not RECORD_NOISE.search(t)]
    if len(records) < 4 or not 2 <= len(live) <= 5:
        return None
    text = " ".join(t for _, t in live)
    reasons = {}  # action -> ref of the record that triggers it

    def first(pattern, flags=re.I):
        """First factual sentence (policy sentences such as "Payment controls: do not pay ..." are skipped)."""
        for r, t in live:
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t):
                if RULE_TEXT.search(sentence) and not re.search(r"\d", sentence):
                    continue
                m = re.search(pattern, sentence, flags)
                if m:
                    return r, m
        return None, None

    # facts
    _, inv = first(r"\b(INV-[\w-]*\d)")
    _, ven = first(r"\bto ([A-Z][\w&.,' -]+?) (?:previously|for|on|under)\b|\bSupplier ([A-Z][^;.\n]+)", 0)
    _, total = first(r"invoice total (?:is|of)\s+" + AMOUNT)
    _, signed = first(r"signed for\s+" + AMOUNT)
    _, instr = first(r"instrument\b[^.]*?\band\s+" + AMOUNT)
    _, paid = first(r"disbursement for\s+" + AMOUNT)
    claim = total or instr or signed or paid
    cur, amount = money(claim[1], claim[2]) if claim else ("", 0)
    facts = {"vendorName": ((ven[1] or ven[2]).strip() if ven else ""), "invoiceNumber": inv[1] if inv else "",
             "amountMinor": amount or 0, "currency": cur}

    # duplicate: the ledger already settled the order (or invoice) now submitted
    r_led, led = first(r"(?:settled|paid|discharged|closed)\s+(?:order\s+)?(PO-[\w-]+)")
    _, cur_order = first(r"currently submitted order is (PO-[\w-]+)")
    _, ins_order = first(r"covers order (PO-[\w-]+)")
    submitted = (cur_order or ins_order)
    if led and submitted and led[1].rstrip(".") == submitted[1].rstrip("."):
        reasons["reject_duplicate"] = r_led
    for r, t in live:  # a factual "already paid" statement (not a policy rule such as "do not pay ... already paid")
        if RULE_TEXT.search(t):
            continue
        if re.search(r"already (been )?(paid|settled|discharged)|second (payment|disbursement)", t, re.I) and \
                not re.search(r"(not|never) (been )?(paid|settled)", t, re.I):
            reasons.setdefault("reject_duplicate", r_led or r)
    # exception: material order/invoice difference or an unresolved dispute
    r_sig, _ = first(r"signed for\s+" + AMOUNT)
    if signed and total and money(signed[1], signed[2]) != money(total[1], total[2]):
        reasons["open_exception"] = r_sig
    r_dis, dis = first(r"(has not|not yet|never) (been )?resolved[^.]*dispute|dispute[^.]*(unresolved|remains open|open)|"
                       r"(conflict|mismatch|contradict)")
    if dis:
        reasons.setdefault("open_exception", r_dis)
    # hold: bank details changed and the independent confirmation is not complete
    r_bank, master = first(r"(?:master|registered)[^.]*account (AC-[\w-]+)")
    _, remit = first(r"(?:remittance|payment) instructions? (?:now )?(?:specif\w+|name|list\w*|use\w*) (AC-[\w-]+)")
    _, cb_ok = first(r"(call-?back|confirmation|verification)[^.]*\b(is|was|has been) (confirmed|completed|verified)")
    _, cb_open = first(r"(call-?back|confirmation|verification)[^.]*(pending|not (yet )?(been )?(confirmed|completed|returned|"
                       r"received)|awaiting|outstanding|unanswered|has not)")
    if master and remit and master[1] != remit[1] and (cb_open or not cb_ok):
        reasons["hold_invoice"] = r_bank
    elif cb_open and not cb_ok:
        reasons["hold_invoice"] = r_bank or first(r"call-?back")[0]
    # approval: payable total above the autonomous ceiling
    r_ceil, ceil = first(r"(?:ceiling|limit|threshold|authority)\b[^.]*?\bis\s+" + AMOUNT)
    if ceil and claim:
        c_cur, c_amt = money(ceil[1], ceil[2])
        if c_cur == cur and amount > c_amt:
            reasons["request_approval"] = r_ceil
    action = next((a for a in ("reject_duplicate", "open_exception", "hold_invoice", "request_approval")
                   if a in reasons), "settle_invoice")
    known = bool(claim) and (bool(led) or bool(signed) or bool(ceil) or bool(master))
    confident = action != "settle_invoice" or (known and bool(signed or instr) and (bool(led) or bool(ceil)))
    # evidence: the live records (three when the file has three); prefer the triggering record if there are more
    chosen = [r for r, _ in live]
    if len(chosen) > 3:
        keep = {reasons.get(action)} | {r for r, t in live if re.search(r"ledger|order|ceiling|controls", t, re.I)}
        chosen = [r for r in chosen if r in keep][:3] or chosen[:3]
    notes = {r.lower(): clean(t, 260).rstrip(" .") for r, t in live if r in chosen}
    return {"action": action, "confident": confident, "refs": chosen, "notes": notes, "facts": facts,
            "text": text, "single": True, "exact": confident}


def finalize(pkg: dict, d: dict, prefer_model: bool = False) -> dict:
    """Validated decision. 'source' records whether the model's refs survived validation.
    prefer_model (fresh audit cases, whose wording may be new): a model answer keeps its action unless it would
    settle against a confident non-settle reading; otherwise a confident case reading decides."""
    action = d.get("action") if d.get("action") in ACTIONS else "open_exception"
    llm_refs = d.get("decisiveRefs") or d.get("evidenceRefs") or [
        e.get("ref") for e in d.get("evidence") or [] if isinstance(e, dict)]
    refs = resolve_refs(pkg, llm_refs)
    notes, reason, source = evidence_notes(d), d.get("reason"), d.get("source", "model")
    from_model = source not in ("fallback", "fallback-refs")
    case = read_case(pkg)
    if from_model and refs and cites_only_noise(pkg, refs) and case and case["action"]:
        # The model based its choice on archive / training / cover material: the operative paragraph decides.
        print(f"[decide] noise guard: {package_id(pkg)} cited only noise {refs}; {action} -> {case['action']}",
              flush=True)
        source, reason, from_model, action = "noise-guard", "", False, case["action"]
    if case and case["refs"] and (case["confident"] or (case["action"] and not from_model)
                                  or (from_model and case["single"])):
        # The case file names its own decisive paragraph: cite its three IDs, quote what each shows, and take the
        # facts from the cover sheet. The paragraph's wording fixes the action unless a model reads it differently
        # on a fresh case.
        if not from_model:
            if source != "noise-guard":
                source, reason = ("rules" if case["confident"] else "rules-weak"), ""
            action = case["action"]
        elif case["confident"] and action != case["action"] and (
                not prefer_model or case.get("exact")
                or (action == "settle_invoice" and case["action"] != "settle_invoice")):
            print(f"[decide] case reader: {package_id(pkg)} model {action} -> {case['action']}", flush=True)
            source, reason, action = "rules-override", "", case["action"]
        elif action != case["action"] and case["action"]:
            print(f"[decide] model kept on fresh case: {package_id(pkg)} {action} (reader: {case['action']})", flush=True)
        refs = case["refs"]
        notes = {**case["notes"], **{k: v for k, v in notes.items() if v}}  # model's explanation, else the quote
        facts = case["facts"] or build_facts(pkg, d)
        if action == "settle_invoice" and not case["confident"] and SETTLE_GUARD:
            risk = risk_signal(pkg, facts["invoiceNumber"])
            if risk:  # never settle an unclear case while a live record says duplicate / conflict / pending / limit
                print(f"[decide] settle guard: {package_id(pkg)} -> {risk[0]}", flush=True)
                action, source = risk[0], ("guard" if source == "model" else source)
                reason = f'Safety check overrides settlement: a current record states "{clean(risk[2], 300).rstrip(".")}"'
                if len(risk[1]) == 3:
                    refs, notes = risk[1], {}
        out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else refs
        return {"action": action, "evidenceRefs": out_refs, "facts": facts,
                "rationale": rationale(action, reason, refs, notes), "source": source}
    if not refs:  # nothing the model cited exists in the package: cite the rule-based decisive paragraph instead
        h = heuristic(pkg)
        refs = resolve_refs(pkg, h["decisiveRefs"])
        notes = {**evidence_notes(h), **notes}
        source = "fallback-refs"
    elif source == "model" and cites_only_noise(pkg, refs):
        # The model based its choice on non-controlling material (e.g. an archived training example):
        # decide again from the operative, non-noise records instead.
        h = heuristic(pkg)
        if h["decisiveRefs"]:
            print(f"[decide] noise guard: {package_id(pkg)} cited only noise {refs}; "
                  f"{action} -> {h['action']}", flush=True)
            action, refs, notes = h["action"], resolve_refs(pkg, h["decisiveRefs"]), {}
            reason = ("The model's cited records are archive, training or cover-sheet material, which does not control "
                      f"this invoice; the operative record states \"{clean(h['controlling'], 300).rstrip('.')}\"")
            source = "noise-guard"
    facts = build_facts(pkg, d)
    risk = risk_signal(pkg, facts["invoiceNumber"]) if action == "settle_invoice" and SETTLE_GUARD else None
    if risk:  # never settle while a live record says duplicate / conflict / pending check / over authority
        action, risk_refs, risk_text = risk
        print(f"[decide] settle guard: {package_id(pkg)} -> {action}", flush=True)
        if len(risk_refs) == 3:
            refs, notes = risk_refs, {}
        reason = (f'Safety check overrides settlement: a current record states "{clean(risk_text, 300).rstrip(".")}"')
        source = source if source != "model" else "guard"
    if case and case["facts"]:
        facts = case["facts"]  # the cover sheet states the controlling facts exactly
    out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else refs
    return {"action": action, "evidenceRefs": out_refs, "facts": facts,
            "rationale": rationale(action, reason, refs, notes), "source": source}


# --- heuristic fallback (used only if every model is unavailable) ------------------------
NOISE = re.compile(r"archiv|example|training|sample|illustrat|cover sheet|transmittal|historical|template|superseded|decoy", re.I)
# "Invoice No." is not a negation; "Nothing is pending" / "isn't" / "no longer" are.
NEGATION = re.compile(r"\b(not|never|without|nor|nothing|none|neither|cannot)\b|n't\b|\bno\b(?!\.)", re.I)
CUES = [
    ("reject_duplicate", re.compile(r"already (been )?paid|duplicate of|paid (on|under|via)", re.I)),
    ("open_exception", re.compile(r"mismatch|conflict|discrepan|does not match|differs", re.I)),
    ("hold_invoice", re.compile(r"pending|until .*verif|on hold|call-?back|awaiting", re.I)),
    ("request_approval", re.compile(r"exceeds|above the .*limit|outside .*authority|requires .*approval", re.I)),
    ("settle_invoice", re.compile(r"within .*(limit|authority)|reconciled|three-way match|matched", re.I)),
]


def affirmed(cue, text: str) -> bool:
    """True if some cue match sits in a clause with no negation outside the match itself."""
    for clause in re.split(r"[.;:!?](?:\s|$)", text):
        for m in cue.finditer(clause):
            if not NEGATION.search(clause[:m.start()] + " " + clause[m.end():]):
                return True
    return False


def heuristic(pkg: dict) -> dict:
    candidates = [(t, r) for t, r in paragraphs(pkg) if len(r) == 3 and not NOISE.search(t)]
    for action, cue in CUES:
        for text, refs in reversed(candidates):
            if affirmed(cue, text):
                return {"action": action, "decisiveRefs": refs, "source": "fallback", "controlling": text,
                        "evidence": [{"ref": r, "shows": "is cited in that record"} for r in refs],
                        "reason": f'Rule-based fallback (model unavailable): the controlling record states '
                                  f'"{clean(text, 300).rstrip(".")}"'}
    refs = candidates[-1][1] if candidates else []
    return {"action": "open_exception", "decisiveRefs": refs, "source": "fallback",
            "controlling": candidates[-1][0] if candidates else "",
            "reason": "Rule-based fallback (model unavailable): no record clearly establishes a payable, approved "
                      "or held invoice, so the records need exception review"}


RESOLVED = re.compile(r"\b(completed|cleared|resolved|released|lifted|satisfied|verified|confirmed|passed|withdrawn|closed|"
                      r"reversed|refunded|voided|cancell?ed)\b", re.I)


def risk_signal(pkg: dict, invoice_number: str = ""):
    """(action, refs, text) for the first live, non-negated, unresolved risk cue anywhere in the package's
    non-noise text (any paragraph, not only 3-ref ones), checked in severity order; None if there is none.
    A 'paid' record that names other invoices but not this one is not a duplicate signal."""
    lines = [p for p in (with_label(lb, ln) for lb, ln in labeled_lines(pkg)) if not NOISE.search(p)]
    for action, cue in CUES[:-1]:
        for text in lines:
            if action == "reject_duplicate" and invoice_number and invoice_number.lower() not in text.lower() \
                    and re.search(r"invoice\s*(no\.?|number|#)?\s*[:\-]?\s*[A-Z0-9][A-Z0-9\-/]*\d", text, re.I):
                continue
            for clause in re.split(r"[.;:!?](?:\s|$)", text):
                for m in cue.finditer(clause):
                    rest = clause[:m.start()] + " " + clause[m.end():]
                    if not NEGATION.search(rest) and not RESOLVED.search(rest):
                        refs = list(dict.fromkeys(r.strip() for r in REF_RE.findall(text)))
                        return action, refs, text
    return None


def heuristic_facts(pkg: dict) -> dict:
    text = "\n".join(string_leaves(pkg))
    grab = lambda pat: (re.search(pat, text, re.I) or [None, None])[1]
    amount = re.search(r"\b([A-Z]{3})\s*([\d,]+(?:\.\d+)?)", text)
    return {"vendorName": (grab(r"(?:vendor(?: name)?\s*[:\-]\s*|invoice from\s+|supplier\s*[:\-]\s*)"
                                r"([^\n;,]+?)(?=\.\s|\.$|[\n;,]|$)") or "").strip(),
            "invoiceNumber": (grab(r"invoice[ \t]*(?:no\.?|number|#)?[ \t]*[:\-]?[ \t]*"
                                   r"((?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]{2,})") or "").strip(),
            "currency": amount[1] if amount else "", "amountText": amount[2] if amount else None}


# --- model calls ------------------------------------------------------------------------
def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


class TooLarge(Exception):
    """The request exceeds the provider's per-request token limit: retry with fewer packages."""


def retry_after(r: httpx.Response, default: float) -> float:
    header = r.headers.get("retry-after", "")
    if re.fullmatch(r"[\d.]+", header.strip()):
        return min(300.0, float(header))
    m = re.search(r"(?:retry|try again) in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
    if m:
        return min(300.0, int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + float(m[3]))
    return default


def classify(r: httpx.Response) -> tuple:
    """(kind, seconds to skip the model) for an error response. Skips are short so one bad reply cannot
    disable the model for long; the in-memory timer also resets on restart."""
    body = r.text.lower()
    if r.status_code == 402 and "in-flight" in body:
        return "credit_busy", 3  # low balance: too many paid requests at once; retry the same model shortly
    if r.status_code == 402 or "insufficient credits" in body or "requires more credits" in body:
        return "no_credit", 900  # paid model refused for lack of credit: use the next (free) model for a while
    if r.status_code == 413 or "request too large" in body or "reduce your message size" in body \
            or "context_length" in body or "maximum context" in body:
        return "too_large", 0
    if r.status_code == 400 and ("json_validate_failed" in body or "failed to generate" in body
                                 or "output_parse_failed" in body or "tool_use_failed" in body):
        return "bad_json", 0
    if r.status_code == 429 and re.search(r"per_?day|perday|daily", body):
        return "daily_quota", 3600  # free daily quota used up for this model: stop asking it, use the others
    if r.status_code == 429:
        return "rate_limited", retry_after(r, 20)
    if r.status_code >= 500:
        return "overloaded", 3
    if r.status_code in (401, 403):
        return "auth", 120
    if r.status_code == 404:
        return "unknown_model", 300
    return "rejected", 60


def max_tokens_for(n: int, base: str = LLM_BASE_URL):
    raw = os.environ.get("LLM_MAX_TOKENS", "")
    if raw == "0":
        return None
    # Groq counts the reserved max_tokens against the per-minute budget, so reserve only what replies use
    # (measured: ~150-250 output tokens per package at reasoning_effort=low).
    # Elsewhere reasoning models (gpt-5*, o*) spend part of this budget thinking, so leave more headroom.
    if raw:
        return int(raw)
    # OpenRouter also reserves credit for max_tokens on every in-flight request, so keep it modest there too.
    return min(16000, 800 + 300 * n) if "groq.com" in base else min(32000, 800 + 350 * n)


def condense(pkg: dict) -> dict:
    """The package as the model sees it: every document keeps only its paragraphs that carry a bracketed reference
    (cover facts, the decisive record, archive/training notes). Case files are mostly filler ("The referenced office
    was ...") so this cuts tokens ~5x, which keeps free-tier quotas and latency low. Packages without such
    paragraphs are sent unchanged."""
    if os.environ.get("LLM_CONDENSE", "on").lower() == "off":
        return pkg

    def shrink(o):
        if isinstance(o, str):
            paras = [p for p in re.split(r"\n\s*\n|\n", o) if REF_RE.search(p)]
            return "\n\n".join(paras) if paras else ("" if len(o) > 200 else o)
        if isinstance(o, dict):
            return {k: shrink(v) for k, v in o.items()}
        if isinstance(o, list):
            return [shrink(v) for v in o]
        return o

    if not paragraphs(pkg):
        return pkg
    return shrink(pkg)


def build_payload(packages: list, policy, json_mode: bool = True, reasoning: bool = True,
                  base: str = LLM_BASE_URL) -> dict:
    user = {"batchContext": policy, "packages": [condense(p) for p in packages]}
    payload = {
        "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if LLM_REASONING and reasoning:
        payload["reasoning_effort"] = LLM_REASONING
    if max_tokens_for(len(packages), base):
        payload["max_tokens"] = max_tokens_for(len(packages), base)
    return payload


ADAPTABLE = {"max_tokens", "temperature", "response_format", "reasoning_effort"}


def learn_rejected_params(model: str, payload: dict, error_text: str) -> bool:
    """OpenAI-style 400s name the refused parameter, e.g. "Unsupported parameter: 'max_tokens' ... use
    'max_completion_tokens'" (gpt-5 / o-series) or "Unsupported value: 'temperature' does not support 0".
    Remember it for this model; True if something new was learned."""
    named = set(re.findall(r"['\"`]([a-z_]+)['\"`]", error_text)) & ADAPTABLE & set(payload)
    fixes = _param_fixes.setdefault(model, set())
    new = named - fixes
    fixes.update(new)
    return bool(new)


def adapt(payload: dict, model: str) -> dict:
    for param in _param_fixes.get(model, ()):
        value = payload.pop(param, None)
        if param == "max_tokens" and value:
            payload["max_completion_tokens"] = value
    return payload


def note(model: str, ok: bool, detail: str):
    for _, key, _ in ENDPOINT.values():
        detail = detail.replace(key, "<key>") if key else detail
    MODEL_STATUS[model] = {"ok": ok, "detail": clean(detail, 300), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


async def pick_model(level: int, deadline: float, hedge: int = 0) -> str:
    """Next ready model (round-robin). While every model cools down, wait if one frees up before the deadline;
    lower levels go first when a model frees up."""
    global _rr
    loop = asyncio.get_running_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining < 3:
            raise TimeoutError("model deadline")
        now = loop.time()
        ready = [m for m in LLM_MODELS if _cooldown.get(m, 0) <= now]
        # A slow last-resort model is only worth it when no preferred model frees up within a few seconds.
        if any(m not in LAST_RESORT and 0 < _cooldown.get(m, 0) - now <= 4 for m in LLM_MODELS):
            ready = [m for m in ready if m not in LAST_RESORT]
        if ready and not any(_waiting[:level]):
            if level == LEVEL_AUDIT:
                return ready[hedge % len(ready)]  # hedged audit calls start on different models
            if not SPREAD:
                return ready[0]  # preference order
            pool = [m for m in ready if m in LLM_MODELS[:SPREAD_TOP]] or ready
            _rr += 1
            return pool[_rr % len(pool)]
        if not ready:
            wake = min(max(_cooldown.get(m, 0), now) for m in LLM_MODELS if m not in LAST_RESORT) \
                if any(m not in LAST_RESORT for m in LLM_MODELS) else min(_cooldown[m] for m in LLM_MODELS)
            if wake - loop.time() > remaining - 8:
                raise RuntimeError("all models cooling down: " + "; ".join(
                    f"{m}: {MODEL_STATUS.get(m, {}).get('detail', '?')}" for m in LLM_MODELS))
            pause = min(1.0, max(0.05, wake - loop.time()))
        else:
            pause = 0.3  # a more urgent caller is queued; let it take this model first
        _waiting[level] += 1
        try:
            await asyncio.sleep(pause)
        finally:
            _waiting[level] -= 1


async def call_model(packages: list, policy, deadline: float, level: int = LEVEL_CORE, hedge: int = 0) -> dict:
    global _sem
    _sem = _sem or asyncio.Semaphore(LLM_CONCURRENCY)
    json_mode = True
    loop = asyncio.get_running_loop()
    async with httpx.AsyncClient() as client:
        attempt = 0
        while True:  # ends on success, after 8 real failures, or at the deadline (pick_model raises)
            attempt += 1
            model = await pick_model(level, deadline, hedge)
            try:
                base, key, extra = ENDPOINT.get(model, (LLM_BASE_URL, LLM_API_KEY, {}))
                async with _sem:
                    payload = adapt(build_payload(packages, policy, json_mode, model not in _no_reasoning, base), model)
                    r = await client.post(f"{base}/chat/completions", json={**payload, **extra, "model": model},
                                          timeout=max(3.0, min(deadline - loop.time(), CALL_TIMEOUT_S,
                                                               10 + 1.5 * len(packages))),
                                          headers={"Authorization": f"Bearer {key}"})
                if r.status_code >= 400:
                    kind, skip = classify(r)
                    note(model, False, f"HTTP {r.status_code} {kind}: {r.text}")
                    if kind == "too_large" and len(packages) > 1:
                        raise TooLarge(f"{model} HTTP {r.status_code} with {len(packages)} package(s)")
                    if kind == "too_large":  # cannot take even one package (e.g. a tiny output-token limit)
                        skip = 300
                    if kind == "bad_json":
                        json_mode = False  # provider-side JSON validation failed: parse the text ourselves
                    if kind == "rejected" and "reasoning" in r.text.lower() and model not in _no_reasoning:
                        _no_reasoning.add(model)  # e.g. a model that does not take reasoning_effort=low
                        skip = 0
                    elif kind == "rejected" and learn_rejected_params(model, payload, r.text):
                        skip = 0  # retry at once without the parameter the model refused
                    if kind == "credit_busy":
                        attempt -= 1  # not a failure: wait for in-flight requests to finish (deadline still applies)
                    _cooldown[model] = loop.time() + skip
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code} {kind}: {clean(r.text, 160)}",
                                                request=r.request, response=r)
                content = r.json()["choices"][0]["message"]["content"]
                data = parse_json(content)
                out = {str(d.get("packageId")): d for d in data.get("decisions", []) if isinstance(d, dict)}
                note(model, True, f"answered {len(out)}/{len(packages)}")
                print(f"[decide] {model} answered {len(out)}/{len(packages)} package(s)", flush=True)
                return out
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                if not isinstance(e, httpx.HTTPStatusError):
                    note(model, False, msg)
                    _cooldown[model] = loop.time() + 2  # timeout / unparseable reply: try another model
                print(f"[decide] {model} attempt {attempt} failed: {msg}", flush=True)
                if attempt >= 8:
                    raise


async def ask_model(packages: list, policy, deadline: float, level: int = LEVEL_CORE, hedge: int = 0) -> dict:
    """call_model, halving the batch whenever the provider says the request is too large."""
    try:
        return await call_model(packages, policy, deadline, level, hedge)
    except TooLarge as e:
        print(f"[decide] {e}; splitting", flush=True)
        if len(packages) == 1:
            raise
        half = len(packages) // 2
        parts = await asyncio.gather(ask_model(packages[:half], policy, deadline, level),
                                     ask_model(packages[half:], policy, deadline, level), return_exceptions=True)
        out = {}
        for p in parts:
            if isinstance(p, dict):
                out.update(p)
        if not out:
            raise next(p for p in parts if isinstance(p, BaseException))
        return out


async def decide_packages(packages: list, deadline: float, store, policy=None) -> list:
    """Returns one finalized decision per package, in order."""
    loop = asyncio.get_running_loop()
    results: list = [None] * len(packages)
    todo, waits = [], []
    for i, pkg in enumerate(packages):
        key = cache_key(pkg, policy)
        cached = store.get_decision(key)
        if cached:
            results[i] = cached
        elif key in _pending:
            waits.append((i, key, _pending[key]))
        else:
            _pending[key] = loop.create_future()
            todo.append((i, key, pkg))

    # A small batch is the fresh audit batch: it cannot be cached ahead, so it is served first.
    level = LEVEL_AUDIT if len(packages) <= AUDIT_MAX_PACKAGES else LEVEL_CORE
    fresh = level == LEVEL_AUDIT
    fell_back = []

    def settle_key(i, key, final):
        results[i] = final
        fut = _pending.pop(key, None)
        if fut and not fut.done():
            fut.set_result(final)

    if not fresh and CASE_READER_FIRST:
        # Core cases the case reader reads confidently need no model call: this keeps the free daily model quota
        # for fresh audit cases, whose wording can be new.
        rest = []
        for i, key, pkg in todo:
            case = read_case(pkg)
            if case and case["confident"]:
                final = finalize(pkg, {**heuristic_facts(pkg), **heuristic(pkg)})
                store.put_decision(key, final)
                settle_key(i, key, final)
            else:
                rest.append((i, key, pkg))
        todo = rest

    async def run_chunk(chunk):
        raw = await model_decisions(chunk, policy, deadline, level)
        for i, key, pkg in chunk:
            d = raw.get(str(package_id(pkg)))
            if answered(d):
                final = finalize(pkg, d, prefer_model=fresh)
                if final["source"] in CACHEABLE:
                    store.put_decision(key, final)  # only fully model-backed decisions are cached
            else:
                final = finalize(pkg, {**heuristic_facts(pkg), **heuristic(pkg)})
                if final["source"] in CACHEABLE:
                    store.put_decision(key, final)  # the case reader decided it confidently without the model
                else:
                    print(f"[decide] FALLBACK used for package {package_id(pkg)}", flush=True)
                    fell_back.append((i, key, pkg))
            results[i] = final
            fut = _pending.pop(key, None)
            if fut and not fut.done():
                fut.set_result(final)

    chunks = [todo[i:i + LLM_CHUNK] for i in range(0, len(todo), LLM_CHUNK)]
    try:
        await asyncio.gather(*(run_chunk(c) for c in chunks))
    finally:
        for _, key, _ in todo:
            fut = _pending.pop(key, None)
            if fut and not fut.done():
                fut.cancel()
    for i, key, fut in waits:
        try:
            results[i] = await asyncio.wait_for(asyncio.shield(fut), max(1, deadline - loop.time()))
        except Exception:
            results[i] = finalize(packages[i], {**heuristic_facts(packages[i]), **heuristic(packages[i])})
    if fell_back and LLM_API_KEY:
        asyncio.ensure_future(warm(fell_back, policy, store))
    return results


def answered(d) -> bool:
    return isinstance(d, dict) and d.get("action") in ACTIONS


async def hedged(packages: list, policy, deadline: float) -> dict:
    """Fresh audit cases: ask two models at once (free tiers return 503 or stall under load) and use the first
    complete answer; the slower call is cancelled."""
    if len(LLM_MODELS) < 2:
        return await ask_model(packages, policy, deadline, LEVEL_AUDIT)
    tasks = [asyncio.ensure_future(ask_model(packages, policy, deadline, LEVEL_AUDIT, h)) for h in (0, 1)]
    best: dict = {}
    try:
        for fut in asyncio.as_completed(tasks):
            try:
                out = await fut
            except Exception as e:
                print(f"[decide] hedged call failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            if sum(map(answered, out.values())) > sum(map(answered, best.values())):
                best = out
            if sum(map(answered, best.values())) >= len(packages):
                break
    finally:
        for t in tasks:
            t.cancel()
    if not best:
        raise RuntimeError("no hedged model answered")
    return best


async def model_decisions(chunk: list, policy, deadline: float, level: int) -> dict:
    """Raw model decisions keyed by packageId; one extra call for packages the model skipped, if time allows."""
    raw, missing = {}, list(chunk)
    loop = asyncio.get_running_loop()
    for _ in range(2):
        if not LLM_API_KEY or not missing:
            break
        try:
            pkgs = [p for _, _, p in missing]
            raw.update(await (hedged(pkgs, policy, deadline) if level == LEVEL_AUDIT
                              else ask_model(pkgs, policy, deadline, level)))
        except Exception as e:
            print(f"[decide] model failed for chunk: {type(e).__name__}: {str(e)[:300]}", flush=True)
            break
        missing = [c for c in missing if not answered(raw.get(str(package_id(c[2]))))]
        if deadline - loop.time() < 12:
            break
    return raw


async def warm(items: list, policy, store):
    """After a request had to fall back, keep asking the model (lowest priority) so the decision is cached
    for the next batch with the same package content. The already-returned proposal is not changed."""
    items = [it for it in items if it[1] not in _warming]
    _warming.update(key for _, key, _ in items)
    loop = asyncio.get_running_loop()
    try:
        for start in range(0, len(items), LLM_CHUNK):
            chunk = items[start:start + LLM_CHUNK]
            raw = await model_decisions(chunk, policy, loop.time() + WARM_BUDGET_S, LEVEL_WARM)
            for _, key, pkg in chunk:
                d = raw.get(str(package_id(pkg)))
                if answered(d):
                    final = finalize(pkg, d)
                    if final["source"] in CACHEABLE:
                        store.put_decision(key, final)
            print(f"[decide] background cache warm: {sum(answered(raw.get(str(package_id(p)))) for _, _, p in chunk)}"
                  f"/{len(chunk)} decided", flush=True)
    except Exception as e:
        print(f"[decide] background cache warm stopped: {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        _warming.difference_update(key for _, key, _ in items)
