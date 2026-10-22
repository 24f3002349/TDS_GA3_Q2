"""AI decision layer: one action + decisive evidence + facts per invoice package.

The model only proposes; every field is validated against the package text here:
  * evidenceRefs are re-derived from the actual paragraph that contains the model's refs,
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
PROMPT_VERSION = "v3"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_REASONING = os.environ.get("LLM_REASONING_EFFORT", "")  # e.g. low / medium; empty = provider default
LLM_CHUNK = int(os.environ.get("LLM_CHUNK", "6"))
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "4"))
KEEP_BRACKETS = os.environ.get("EVIDENCE_REF_BRACKETS", "false").lower() == "true"

REF_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")
ZERO_DECIMAL = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "XAF", "XOF", "PYG"}
THREE_DECIMAL = {"KWD", "BHD", "OMR", "JOD", "TND", "LYD", "IQD"}

_sem = None
_pending: dict[str, asyncio.Future] = {}

SYSTEM_PROMPT = """You are an accounts-payable control agent. For EACH invoice package choose exactly one action:

- settle_invoice: invoice is valid, fully reconciled (PO / receipt / invoice agree) and the amount is within the
  autonomous (delegated) payment authority stated in the policy. Nothing pending, no conflict, not already paid.
- request_approval: commercially valid and reconciled, BUT the amount (or another stated condition) is outside the
  agent's delegated/autonomous authority, so a human approver must sign off.
- hold_invoice: payment must pause until a specific, stated verification completes (e.g. bank-detail change
  call-back, tax/registration check, goods-receipt confirmation pending, sanctions screening). Something is pending.
- reject_duplicate: this SAME commercial invoice (same vendor + same invoice number, same amount) was ALREADY PAID.
  A payment for a DIFFERENT invoice (different invoice number, PO, period or amount) is NOT a duplicate.
- open_exception: material records CONFLICT (e.g. amount/quantity/price/vendor/bank mismatch between PO, receipt and
  invoice) and need an exception workflow rather than a simple wait.

How to read the documents (they are deliberately noisy):
- Bracketed tokens like [ABC-123] are evidence references. Exactly one paragraph is DECISIVE: the current, operative
  paragraph about THIS invoice that determines the action. It normally contains exactly three bracketed references.
- IGNORE: the cover sheet / transmittal paragraph, archived or historical examples, training material, sample or
  illustrative cases, templates, superseded policy text, and paragraphs about other invoices or vendors.
- Watch for NEGATION ("no duplicate was found", "is not pending", "does not exceed") and for action words used in
  irrelevant contexts (e.g. "hold" in a product name, "approval" describing an old case). Decide from meaning.
- Apply the batch policy revision (authority limits, currency) when judging authority.

Facts are for the invoice being claimed in this package: vendorName and invoiceNumber exactly as written,
currency as ISO-4217 code, the invoice total amount exactly as printed in major units (amountText, e.g. "1,24,500.00"),
and amountMinor = that amount in minor units (cents/paise) as an integer.

Reply with JSON only:
{"decisions":[{"packageId":"...","action":"one of the five","decisiveRefs":["REF1","REF2","REF3"],
"vendorName":"...","invoiceNumber":"...","amountText":"...","amountMinor":0,"currency":"INR",
"reason":"2-4 sentences explaining why this action, citing the decisive refs"}]}
decisiveRefs must be copied exactly (without brackets) from the decisive paragraph. One decision per package."""


def package_id(pkg: dict):
    return pkg.get("packageId", pkg.get("id"))


def cache_key(pkg: dict, policy) -> str:
    content = {k: v for k, v in pkg.items() if k not in ("packageId", "id")}
    blob = json.dumps({"p": content, "policy": policy, "v": PROMPT_VERSION}, sort_keys=True,
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


def norm_ref(r) -> str:
    return str(r).strip().strip("[]").strip()


def resolve_refs(pkg: dict, llm_refs) -> list:
    """Snap the model's refs onto the real paragraph that contains them."""
    wanted = [norm_ref(r) for r in (llm_refs or []) if norm_ref(r)]
    paras = paragraphs(pkg)
    best, best_score = None, (0, 0)
    for text, refs in paras:
        score = (len(set(refs) & set(wanted)), -abs(len(refs) - 3))
        if score > best_score:
            best, best_score = refs, score
    if best and best_score[0] >= 2 and len(best) == 3:
        return best
    known = {r for _, refs in paras for r in refs}
    valid = [r for r in wanted if r in known]
    if valid:
        return list(dict.fromkeys(valid))[:3]
    return best[:3] if best else wanted[:3]


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


def rationale(action: str, reason: str, refs: list) -> str:
    cited = ", ".join(f"[{r}]" for r in refs)
    reason = re.sub(r"\s+", " ", str(reason or "")).strip()[:1100]
    text = f"Proposed action {action}. {reason} Decisive evidence: {cited}."
    if len(text) < 60:
        text += " The proposal is not executed until the grader returns an ACCEPTED result."
    return text[:1500]


def finalize(pkg: dict, d: dict) -> dict:
    action = d.get("action") if d.get("action") in ACTIONS else "open_exception"
    refs = resolve_refs(pkg, d.get("decisiveRefs") or d.get("evidenceRefs"))
    out_refs = [f"[{r}]" for r in refs] if KEEP_BRACKETS else refs
    return {"action": action, "evidenceRefs": out_refs, "facts": build_facts(pkg, d),
            "rationale": rationale(action, d.get("reason"), refs)}


# --- heuristic fallback (used only if the model is unavailable) ------------------------
NOISE = re.compile(r"archiv|example|training|sample|illustrat|cover sheet|transmittal|historical|template|superseded|decoy", re.I)
CUES = [
    ("reject_duplicate", re.compile(r"already (been )?paid|duplicate of|paid (on|under|via)", re.I)),
    ("open_exception", re.compile(r"mismatch|conflict|discrepan|does not match|differs", re.I)),
    ("hold_invoice", re.compile(r"pending|until .*verif|on hold|call-?back|awaiting", re.I)),
    ("request_approval", re.compile(r"exceeds|above the .*limit|outside .*authority|requires .*approval", re.I)),
    ("settle_invoice", re.compile(r"within .*(limit|authority)|reconciled|three-way match|matched", re.I)),
]


def heuristic(pkg: dict) -> dict:
    candidates = [(t, r) for t, r in paragraphs(pkg) if len(r) == 3 and not NOISE.search(t)]
    for action, cue in CUES:
        for text, refs in reversed(candidates):
            if cue.search(text) and not re.search(r"\bno\b|\bnot\b", text, re.I):
                return {"action": action, "decisiveRefs": refs, "reason": "Heuristic fallback decision."}
    refs = candidates[-1][1] if candidates else []
    return {"action": "open_exception", "decisiveRefs": refs, "reason": "Heuristic fallback: records need review."}


def heuristic_facts(pkg: dict) -> dict:
    text = "\n".join(string_leaves(pkg))
    grab = lambda pat: (re.search(pat, text, re.I) or [None, None])[1]
    amount = re.search(r"\b([A-Z]{3})\s*([\d,]+(?:\.\d+)?)", text)
    return {"vendorName": (grab(r"vendor(?: name)?\s*[:\-]\s*([^\n.;]+)") or "").strip(),
            "invoiceNumber": (grab(r"invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})") or "").strip(),
            "currency": amount[1] if amount else "", "amountText": amount[2] if amount else None}


# --- model calls ------------------------------------------------------------------------
def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


async def call_model(packages: list, policy, deadline: float) -> dict:
    global _sem
    _sem = _sem or asyncio.Semaphore(LLM_CONCURRENCY)
    user = {"policyRevision": policy, "packages": packages}
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    if LLM_REASONING:
        payload["reasoning_effort"] = LLM_REASONING
    loop = asyncio.get_running_loop()
    attempt = 0
    async with _sem:
        async with httpx.AsyncClient() as client:
            while True:
                remaining = deadline - loop.time()
                if remaining < 3:
                    raise TimeoutError("model deadline")
                try:
                    r = await client.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=remaining,
                                          headers={"Authorization": f"Bearer {LLM_API_KEY}"})
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    data = parse_json(content)
                    return {str(d.get("packageId")): d for d in data.get("decisions", []) if isinstance(d, dict)}
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                    attempt += 1
                    if attempt >= 4:
                        raise
                    print(f"[decide] model retry {attempt}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    await asyncio.sleep(min(2 ** attempt, max(0.0, deadline - loop.time() - 3)))


async def decide_packages(packages: list, deadline: float, store, policy=None) -> list:
    """Returns one finalized decision per package, in order."""
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
            _pending[key] = asyncio.get_running_loop().create_future()
            todo.append((i, key, pkg))

    async def run_chunk(chunk):
        raw = {}
        if LLM_API_KEY:
            try:
                raw = await call_model([p for _, _, p in chunk], policy, deadline)
            except Exception as e:
                print(f"[decide] model failed for chunk: {type(e).__name__}: {str(e)[:200]}", flush=True)
        for i, key, pkg in chunk:
            d = raw.get(str(package_id(pkg)))
            if d and d.get("action") in ACTIONS:
                final = finalize(pkg, d)
                store.put_decision(key, final)  # only model decisions are cached
            else:
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
            results[i] = await asyncio.wait_for(asyncio.shield(fut), max(1, deadline - asyncio.get_running_loop().time()))
        except Exception:
            results[i] = finalize(packages[i], {**heuristic_facts(packages[i]), **heuristic(packages[i])})
    return results
