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
PROMPT_VERSION = "v5"  # bump whenever prompt or validation changes, so stale cached decisions are not reused

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
# LLM_MODEL may be a comma-separated preference list; LLM_FALLBACK_MODELS is appended. When a model answers
# 404 (retired), 429 (quota) or 5xx (overloaded) the next one is tried instead of falling back to keywords.
DEFAULT_MODELS = "gemini-3.6-flash,gemini-3.5-flash,gemini-3.8-flash"
LLM_MODELS = list(dict.fromkeys(
    m.strip() for m in f'{os.environ.get("LLM_MODEL") or DEFAULT_MODELS},{os.environ.get("LLM_FALLBACK_MODELS", "")}'
    .split(",") if m.strip()))
LLM_REASONING = os.environ.get("LLM_REASONING_EFFORT", "")  # e.g. low / medium; empty = provider default
LLM_CHUNK = int(os.environ.get("LLM_CHUNK", "6"))
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "4"))
KEEP_BRACKETS = os.environ.get("EVIDENCE_REF_BRACKETS", "false").lower() == "true"
AUDIT_MAX_PACKAGES = int(os.environ.get("AUDIT_MAX_PACKAGES", "8"))  # batches this small get model priority
WARM_BUDGET_S = float(os.environ.get("WARM_BUDGET_S", "240"))
SETTLE_GUARD = os.environ.get("SETTLE_GUARD", "on").lower() != "off"

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
   When a later or amended record replaces an earlier one, the later record controls.
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


def paragraphs(pkg: dict):
    """All text paragraphs that carry bracketed refs, as (text, [refs in order])."""
    out = []
    for s in string_leaves(pkg):
        for para in re.split(r"\n+", s):
            refs = [r.strip() for r in REF_RE.findall(para)]
            if refs:
                out.append((para, list(dict.fromkeys(refs))))
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


def find_key(obj, names):
    """Breadth-first search for the shallowest key with one of the given names."""
    queue = [obj]
    while queue:
        nxt = []
        for o in queue:
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in names and not isinstance(v, (dict, list)) and v not in (None, ""):
                        return v
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        queue = nxt
    return None


def to_minor(amount_text, currency: str):
    if amount_text is None:
        return None
    m = re.findall(r"\d[\d,'\s]*(?:\.\d+)?", str(amount_text))
    if not m:
        return None
    try:
        value = Decimal(re.sub(r"[,'\s]", "", m[-1]))
    except InvalidOperation:
        return None
    exp = 0 if currency in ZERO_DECIMAL else 3 if currency in THREE_DECIMAL else 2
    return int((value * (10 ** exp)).to_integral_value())


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


def finalize(pkg: dict, d: dict) -> dict:
    """Validated decision. 'source' records whether the model's refs survived validation."""
    action = d.get("action") if d.get("action") in ACTIONS else "open_exception"
    llm_refs = d.get("decisiveRefs") or d.get("evidenceRefs") or [
        e.get("ref") for e in d.get("evidence") or [] if isinstance(e, dict)]
    refs = resolve_refs(pkg, llm_refs)
    notes, reason, source = evidence_notes(d), d.get("reason"), d.get("source", "model")
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
    out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else refs
    return {"action": action, "evidenceRefs": out_refs, "facts": facts,
            "rationale": rationale(action, reason, refs, notes), "source": source}


# --- heuristic fallback (used only if every model is unavailable) ------------------------
NOISE = re.compile(r"archiv|example|training|sample|illustrat|cover sheet|transmittal|historical|template|superseded|decoy", re.I)
NEGATION = re.compile(r"\b(not|never|without|nor)\b|\bno\b(?!\.)", re.I)  # "Invoice No." is not a negation
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


RESOLVED = re.compile(r"\b(completed|cleared|resolved|released|lifted|satisfied|verified|passed|withdrawn|closed|"
                      r"reversed|refunded|voided|cancell?ed)\b", re.I)


def risk_signal(pkg: dict, invoice_number: str = ""):
    """(action, refs, text) for the first live, non-negated, unresolved risk cue anywhere in the package's
    non-noise text (any paragraph, not only 3-ref ones), checked in severity order; None if there is none.
    A 'paid' record that names other invoices but not this one is not a duplicate signal."""
    lines = [p for s in string_leaves(pkg) for p in re.split(r"\n+", s) if p.strip() and not NOISE.search(p)]
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
    return {"vendorName": (grab(r"vendor(?: name)?\s*[:\-]\s*([^\n.;]+)") or "").strip(),
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
    if r.status_code == 413 or "request too large" in body or "reduce your message size" in body \
            or "context_length" in body or "maximum context" in body:
        return "too_large", 0
    if r.status_code == 400 and ("json_validate_failed" in body or "failed to generate" in body
                                 or "output_parse_failed" in body or "tool_use_failed" in body):
        return "bad_json", 0
    if r.status_code == 429:
        return "rate_limited", retry_after(r, 20)
    if r.status_code >= 500:
        return "overloaded", 3
    if r.status_code in (401, 403):
        return "auth", 120
    if r.status_code == 404:
        return "unknown_model", 300
    return "rejected", 60


def max_tokens_for(n: int):
    raw = os.environ.get("LLM_MAX_TOKENS", "")
    if raw == "0":
        return None
    # Groq counts the reserved max_tokens against the per-minute budget, so reserve only what replies use
    # (measured: ~150-250 output tokens per package at reasoning_effort=low).
    return int(raw) if raw else min(16000, 800 + 300 * n)


def build_payload(packages: list, policy, json_mode: bool = True, reasoning: bool = True) -> dict:
    user = {"batchContext": policy, "packages": packages}
    payload = {
        "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if LLM_REASONING and reasoning:
        payload["reasoning_effort"] = LLM_REASONING
    if max_tokens_for(len(packages)):
        payload["max_tokens"] = max_tokens_for(len(packages))
    return payload


def note(model: str, ok: bool, detail: str):
    detail = detail.replace(LLM_API_KEY, "<key>") if LLM_API_KEY else detail
    MODEL_STATUS[model] = {"ok": ok, "detail": clean(detail, 300), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


async def pick_model(level: int, deadline: float) -> str:
    """Next ready model (round-robin). While every model cools down, wait if one frees up before the deadline;
    lower levels go first when a model frees up."""
    global _rr
    loop = asyncio.get_running_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining < 3:
            raise TimeoutError("model deadline")
        ready = [m for m in LLM_MODELS if _cooldown.get(m, 0) <= loop.time()]
        if ready and not any(_waiting[:level]):
            _rr += 1
            return ready[_rr % len(ready)]
        if not ready:
            wake = min(_cooldown[m] for m in LLM_MODELS)
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


async def call_model(packages: list, policy, deadline: float, level: int = LEVEL_CORE) -> dict:
    global _sem
    _sem = _sem or asyncio.Semaphore(LLM_CONCURRENCY)
    json_mode = True
    loop = asyncio.get_running_loop()
    async with httpx.AsyncClient() as client:
        for attempt in range(1, 9):
            model = await pick_model(level, deadline)
            try:
                async with _sem:
                    payload = build_payload(packages, policy, json_mode, model not in _no_reasoning)
                    r = await client.post(f"{LLM_BASE_URL}/chat/completions", json={**payload, "model": model},
                                          timeout=max(3.0, deadline - loop.time()),
                                          headers={"Authorization": f"Bearer {LLM_API_KEY}"})
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
    raise RuntimeError("model attempts exhausted")


async def ask_model(packages: list, policy, deadline: float, level: int = LEVEL_CORE) -> dict:
    """call_model, halving the batch whenever the provider says the request is too large."""
    try:
        return await call_model(packages, policy, deadline, level)
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
    fell_back = []

    async def run_chunk(chunk):
        raw = await model_decisions(chunk, policy, deadline, level)
        for i, key, pkg in chunk:
            d = raw.get(str(package_id(pkg)))
            if answered(d):
                final = finalize(pkg, d)
                if final["source"] == "model":
                    store.put_decision(key, final)  # only fully model-backed decisions are cached
            else:
                print(f"[decide] FALLBACK used for package {package_id(pkg)}", flush=True)
                final = finalize(pkg, {**heuristic_facts(pkg), **heuristic(pkg)})
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


async def model_decisions(chunk: list, policy, deadline: float, level: int) -> dict:
    """Raw model decisions keyed by packageId; one extra call for packages the model skipped, if time allows."""
    raw, missing = {}, list(chunk)
    loop = asyncio.get_running_loop()
    for _ in range(2):
        if not LLM_API_KEY or not missing:
            break
        try:
            raw.update(await ask_model([p for _, _, p in missing], policy, deadline, level))
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
                    if final["source"] == "model":
                        store.put_decision(key, final)
            print(f"[decide] background cache warm: {sum(answered(raw.get(str(package_id(p)))) for _, _, p in chunk)}"
                  f"/{len(chunk)} decided", flush=True)
    except Exception as e:
        print(f"[decide] background cache warm stopped: {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        _warming.difference_update(key for _, key, _ in items)
