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
from decimal import Decimal, InvalidOperation

import httpx

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
PROMPT_VERSION = "v4"

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

REF_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")
ID_KEY_RE = re.compile(r"^(id|ref|evidenceid|evidenceref|refid|docid|documentid|recordid|sourceid)$", re.I)
ZERO_DECIMAL = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "XAF", "XOF", "PYG"}
THREE_DECIMAL = {"KWD", "BHD", "OMR", "JOD", "TND", "LYD", "IQD"}

_sem = None
_pending: dict[str, asyncio.Future] = {}
_cooldown: dict[str, float] = {}  # model -> loop time until which it is skipped

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
    best, best_score = None, (0, 0)
    for _, refs in paras:
        score = (len(set(refs) & set(valid)), -abs(len(refs) - 3))
        if score > best_score:
            best, best_score = refs, score
    if best and best_score[0] >= 2 and len(best) == 3:
        return best
    return valid[:3]


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
    out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else refs
    return {"action": action, "evidenceRefs": out_refs, "facts": build_facts(pkg, d),
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
                return {"action": action, "decisiveRefs": refs, "source": "fallback",
                        "evidence": [{"ref": r, "shows": "is cited in that record"} for r in refs],
                        "reason": f'Rule-based fallback (model unavailable): the controlling record states '
                                  f'"{clean(text, 300).rstrip(".")}"'}
    refs = candidates[-1][1] if candidates else []
    return {"action": "open_exception", "decisiveRefs": refs, "source": "fallback",
            "reason": "Rule-based fallback (model unavailable): no record clearly establishes a payable, approved "
                      "or held invoice, so the records need exception review"}


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


def cooldown_for(r: httpx.Response) -> float:
    """How long to skip a model after an error response."""
    if r.status_code == 404:
        return 86400  # retired / unknown model id
    if r.status_code == 429:
        m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
        if m:
            return min(3600.0, int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + float(m[3]))
        return 60
    if r.status_code >= 500:
        return 5
    return 600  # other 4xx: this model rejects the request shape


async def call_model(packages: list, policy, deadline: float) -> dict:
    global _sem
    _sem = _sem or asyncio.Semaphore(LLM_CONCURRENCY)
    user = {"batchContext": policy, "packages": packages}
    payload = {
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    if LLM_REASONING:
        payload["reasoning_effort"] = LLM_REASONING
    loop = asyncio.get_running_loop()
    async with _sem:
        async with httpx.AsyncClient() as client:
            for attempt in range(1, 7):
                remaining = deadline - loop.time()
                if remaining < 3:
                    raise TimeoutError("model deadline")
                ready = [m for m in LLM_MODELS if _cooldown.get(m, 0) <= loop.time()]
                if not ready:  # every model is cooling down: wait only if one frees up in time
                    wake = min(_cooldown[m] for m in LLM_MODELS)
                    if wake - loop.time() > remaining - 8:
                        raise RuntimeError("all models unavailable (quota/overload): " + ", ".join(LLM_MODELS))
                    await asyncio.sleep(max(0.0, wake - loop.time()))
                    continue
                model = ready[0]
                try:
                    r = await client.post(f"{LLM_BASE_URL}/chat/completions", json={**payload, "model": model},
                                          timeout=remaining, headers={"Authorization": f"Bearer {LLM_API_KEY}"})
                    if r.status_code >= 400:
                        _cooldown[model] = loop.time() + cooldown_for(r)
                        raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {clean(r.text, 160)}",
                                                    request=r.request, response=r)
                    content = r.json()["choices"][0]["message"]["content"]
                    data = parse_json(content)
                    out = {str(d.get("packageId")): d for d in data.get("decisions", []) if isinstance(d, dict)}
                    print(f"[decide] {model} answered {len(out)}/{len(packages)} package(s)", flush=True)
                    return out
                except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as e:
                    print(f"[decide] {model} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    if attempt >= 6:
                        raise
                    if not isinstance(e, httpx.HTTPStatusError):  # timeout / bad JSON: brief pause, same order
                        await asyncio.sleep(min(2 ** attempt, max(0.0, deadline - loop.time() - 3)))
    raise RuntimeError("model attempts exhausted")


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

    def answered(d):
        return isinstance(d, dict) and d.get("action") in ACTIONS

    async def run_chunk(chunk):
        raw, missing = {}, list(chunk)
        for _ in range(2):  # one extra call for packages the model skipped, if time allows
            if not LLM_API_KEY or not missing:
                break
            try:
                raw.update(await call_model([p for _, _, p in missing], policy, deadline))
            except Exception as e:
                print(f"[decide] model failed for chunk: {type(e).__name__}: {str(e)[:200]}", flush=True)
                break
            missing = [c for c in missing if not answered(raw.get(str(package_id(c[2]))))]
            if deadline - loop.time() < 12:
                break
        for i, key, pkg in chunk:
            d = raw.get(str(package_id(pkg)))
            if answered(d):
                final = finalize(pkg, d)
                if final["source"] == "model":
                    store.put_decision(key, final)  # only fully model-backed decisions are cached
            else:
                print(f"[decide] FALLBACK used for package {package_id(pkg)}", flush=True)
                final = finalize(pkg, {**heuristic_facts(pkg), **heuristic(pkg)})
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
    return results
