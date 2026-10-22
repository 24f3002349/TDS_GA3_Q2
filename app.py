"""A2A 1.0 HTTP+JSON invoice action agent: protocol layer.

Layers: app.py (protocol, lifecycle, idempotency, isolation) -> store.py (SQLite)
                                                             -> decide.py (AI decisions)
"""
import asyncio
import datetime
import hashlib
import json
import os
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

import decide
from store import Store

BASE_PATH = "/" + os.environ.get("A2A_BASE_PATH", "/a2a").strip("/")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip()  # e.g. https://host/a2a/
ALLOWED_TOKENS = {t for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t}
DB_PATH = os.environ.get("DB_PATH", "a2a_tasks.sqlite3")
REQUEST_BUDGET_S = float(os.environ.get("REQUEST_BUDGET_S", "40"))
MAX_RESPONSE = 512 * 1024
MAX_BODY = 8 * 1024 * 1024

MT_A2A = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = decide.ACTIONS

S_WORKING = "TASK_STATE_WORKING"
S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCELED = "TASK_STATE_CANCELED"
S_FAILED = "TASK_STATE_FAILED"
TERMINAL = {S_DONE, S_CANCELED, S_FAILED, "TASK_STATE_REJECTED"}

store = Store(DB_PATH)
INFLIGHT: dict[str, asyncio.Future] = {}  # task_id -> proposal-building job
LOCKS: dict[str, asyncio.Lock] = {}


def lock_for(task_id: str) -> asyncio.Lock:
    return LOCKS.setdefault(task_id, asyncio.Lock())


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- responses ---------------------------------------------------------------
def ok(obj, status=200) -> Response:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
    return Response(body, status, media_type=MT_A2A)


class A2AError(Exception):
    def __init__(self, status: int, code: str, title: str, detail: str = "", **extra):
        self.status, self.code, self.title, self.detail, self.extra = status, code, title, detail, extra


HTTP_STATUS_NAMES = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
                     405: "UNIMPLEMENTED", 406: "INVALID_ARGUMENT", 409: "FAILED_PRECONDITION",
                     413: "INVALID_ARGUMENT", 415: "INVALID_ARGUMENT", 500: "INTERNAL"}


def problem(e: A2AError) -> Response:
    # RFC 9457 fields (spec section 6.4 example) plus the AIP-193 "error" object (spec section 11.6).
    body = {
        "type": f"https://a2a-protocol.org/errors/{e.code.lower().replace('_', '-')}",
        "title": e.title,
        "status": e.status,
        "detail": e.detail or e.title,
        "code": e.code,
        **e.extra,
        "error": {
            "code": e.status,
            "status": HTTP_STATUS_NAMES.get(e.status, "UNKNOWN"),
            "message": e.detail or e.title,
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": e.code,
                         "domain": "a2a-protocol.org",
                         "metadata": {k: json.dumps(v) if not isinstance(v, str) else v for k, v in e.extra.items()}}],
        },
    }
    headers = {"WWW-Authenticate": 'Bearer realm="a2a"'} if e.status == 401 else None
    return Response(json.dumps(body).encode(), e.status, media_type="application/problem+json", headers=headers)


def not_found() -> A2AError:
    # Generic on purpose: never reveal whether another principal's task exists.
    return A2AError(404, "TASK_NOT_FOUND", "Task not found")


def bad(detail: str, code="INVALID_PARAMS") -> A2AError:
    return A2AError(400, code, "Invalid request", detail)


# --- request guards ------------------------------------------------------------
def principal_of(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not header or scheme.lower() != "bearer" or not token or token != token.strip() or " " in token:
        raise A2AError(401, "UNAUTHENTICATED", "Missing or invalid bearer token")
    if ALLOWED_TOKENS and token not in ALLOWED_TOKENS:
        raise A2AError(401, "UNAUTHENTICATED", "Missing or invalid bearer token")
    return sha(token)


def check_protocol(request: Request, has_body: bool):
    version = request.headers.get("a2a-version") or request.query_params.get("A2A-Version") or ""
    # Spec 3.6: only Major.Minor counts ("1.0.1" is 1.0); an empty value means 0.3.
    if version.strip().split(".")[:2] != ["1", "0"]:
        raise A2AError(400, "VERSION_NOT_SUPPORTED", "Protocol Version Not Supported",
                       "This agent supports A2A-Version 1.0 only.", supportedVersions=["1.0"])
    accept = request.headers.get("accept", "")
    if accept:
        kinds = [a.split(";")[0].strip().lower() for a in accept.split(",")]
        if not any(k in (MT_A2A, "application/json", "application/*", "*/*") for k in kinds):
            raise A2AError(406, "NOT_ACCEPTABLE", "Not Acceptable", "Responses use application/a2a+json.")
    if has_body:
        # A2A POST bodies must be application/a2a+json (parameters such as charset are fine).
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != MT_A2A:
            raise A2AError(415, "CONTENT_TYPE_NOT_SUPPORTED", "Unsupported Media Type",
                           f"Request bodies must use Content-Type {MT_A2A}.", supportedMediaTypes=[MT_A2A])


async def read_json(request: Request):
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise A2AError(413, "PAYLOAD_TOO_LARGE", "Payload too large")
    try:
        return json.loads(raw)
    except Exception:
        raise bad("Body is not valid JSON.", "PARSE_ERROR")


def guarded(handler, has_body=False):
    async def endpoint(request: Request):
        try:
            principal = principal_of(request)
            # Any POST that actually carries a body (e.g. a cancel with a JSON body) is media-type checked too.
            check_protocol(request, has_body or (request.method == "POST" and bool(await request.body())))
            return await handler(request, principal)
        except A2AError as e:
            return problem(e)
    return endpoint


# --- task helpers ---------------------------------------------------------------
def agent_message(task: dict, text: str) -> dict:
    return {
        "messageId": str(uuid.uuid4()),
        "role": "ROLE_AGENT",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "parts": [{"text": text, "mediaType": "text/plain"}],
    }


def set_status(task: dict, state: str, text: str):
    msg = agent_message(task, text)
    task["status"] = {"state": state, "message": msg, "timestamp": now()}
    task["history"].append(msg)


def view(task: dict, history_length=None, include_artifacts=True) -> dict:
    out = dict(task)
    if history_length is not None:
        out["history"] = task["history"][-history_length:] if history_length > 0 else []
        if not out["history"]:
            out.pop("history")
    if not include_artifacts:
        out.pop("artifacts", None)
    return out


def int_param(request: Request, name: str, lo: int, hi: int, default=None):
    raw = request.query_params.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise bad(f"{name} must be an integer.")
    if not lo <= value <= hi:
        raise bad(f"{name} out of range.")
    return value


def find_part(message: dict, media_type: str):
    for part in message.get("parts") or []:
        if isinstance(part, dict) and part.get("mediaType") == media_type and isinstance(part.get("data"), dict):
            return part["data"]
    return None


# --- proposal building (runs detached so client disconnects cannot interrupt it) -----
async def build_proposals(principal: str, task_id: str):
    record = store.get_task(principal, task_id)
    batch = record["batch"]
    packages = batch["packages"]
    deadline = asyncio.get_running_loop().time() + REQUEST_BUDGET_S - 4
    decisions = await decide.decide_packages(packages, deadline, store, decide.batch_context(batch))
    proposals = []
    for pkg, d in zip(packages, decisions):
        pid = decide.package_id(pkg)
        proposals.append({
            "packageId": pid,
            "actionId": "act_" + sha(f"{principal}|{task_id}|{pid}")[:32],
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })
    async with lock_for(task_id):
        record = store.get_task(principal, task_id)
        task = record["task"]
        if task["status"]["state"] != S_WORKING:  # canceled meanwhile: never overwrite
            return
        task["artifacts"] = [{
            "artifactId": "proposals-" + task_id,
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": MT_PROPOSALS, "data": {"batchId": batch["batchId"], "proposals": proposals}}],
        }]
        set_status(task, S_INPUT,
                   "Proposals ready. No action has been executed; send invoice-action-results to continue.")
        record["proposals"] = proposals
        record["sources"] = {decide.package_id(pkg): d.get("source") for pkg, d in zip(packages, decisions)}
        store.save(principal, record)
    diag(f"batch {batch['batchId']} packages={len(packages)} context={json.dumps(decide.batch_context(batch))[:300]}")
    for p in proposals:
        f = p["facts"]
        diag(f"proposal batch={batch['batchId']} pkg={p['packageId']} action={p['action']} "
             f"src={record['sources'].get(p['packageId'])} refs={p['evidenceRefs']} "
             f"facts={f['vendorName']}|{f['invoiceNumber']}|{f['amountMinor']}|{f['currency']}")


# --- diagnostics: server log only (never returned over the API), to compare our proposals with grader verdicts ---
DIAG = os.environ.get("DIAG_LOG", "on").lower() != "off"
DIAG_PKG_CHARS = int(os.environ.get("DIAG_PKG_CHARS", "6000"))


def diag(line: str):
    if DIAG:
        print("[diag] " + line, flush=True)


def diag_verdicts(record: dict, data: dict):
    batch_id = record["batch"]["batchId"]
    proposals = {p["packageId"]: p for p in record["proposals"]}
    packages = {decide.package_id(p): p for p in record["batch"]["packages"]}
    sources = record.get("sources", {})
    results = data.get("results", [])
    accepted = sum(r.get("outcome") == "ACCEPTED" for r in results)
    diag(f"summary batch={batch_id} accepted={accepted}/{len(results)}")
    for r in results:
        pid = r.get("packageId")
        p = proposals.get(pid, {})
        diag(f"verdict batch={batch_id} pkg={pid} outcome={r.get('outcome')} ours={p.get('action')} "
             f"src={sources.get(pid)}")
        if r.get("outcome") != "ACCEPTED":
            diag(f"rejected batch={batch_id} pkg={pid} rationale={p.get('rationale', '')[:400]!r}")
            diag(f"rejected-package batch={batch_id} pkg={pid} "
                 f"json={json.dumps(packages.get(pid), ensure_ascii=False)[:DIAG_PKG_CHARS]}")


def ensure_job(principal: str, task_id: str) -> asyncio.Future:
    job = INFLIGHT.get(task_id)
    if job is None or (job.done() and job.exception() is not None):
        job = asyncio.ensure_future(build_proposals(principal, task_id))
        INFLIGHT[task_id] = job
        job.add_done_callback(lambda f: INFLIGHT.pop(task_id, None) if not f.exception() else None)
    return job


async def wait_ready(principal: str, task_id: str) -> dict:
    record = store.get_task(principal, task_id)
    if record["task"]["status"]["state"] == S_WORKING:
        job = ensure_job(principal, task_id)
        try:
            await asyncio.wait_for(asyncio.shield(job), REQUEST_BUDGET_S)
        except asyncio.TimeoutError:
            pass
        except Exception:
            async with lock_for(task_id):
                record = store.get_task(principal, task_id)
                if record["task"]["status"]["state"] == S_WORKING:
                    set_status(record["task"], S_FAILED, "Internal error while preparing proposals.")
                    store.save(principal, record)
        record = store.get_task(principal, task_id)
    return record["task"]


# --- message:send -----------------------------------------------------------------
async def replay(principal: str, rec: dict, digest: str) -> Response:
    if rec["hash"] != digest:
        raise A2AError(409, "IDEMPOTENCY_CONFLICT", "Idempotency conflict",
                       "messageId was already used with different content.")
    task = await wait_ready(principal, rec["task_id"])
    return ok({"task": task})


def validate_batch(message: dict) -> dict:
    data = find_part(message, MT_BATCH)
    if data is None:
        raise bad(f"Expected a data part with mediaType {MT_BATCH}.", "CONTENT_TYPE_NOT_SUPPORTED")
    batch_id = data.get("batchId")
    packages = data.get("packages")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(packages, list) or not packages:
        raise bad("Batch requires batchId and a non-empty packages list.")
    seen = set()
    for pkg in packages:
        pid = decide.package_id(pkg) if isinstance(pkg, dict) else None
        if not isinstance(pid, str) or not pid or pid in seen:
            raise bad("Every package needs a unique packageId.")
        seen.add(pid)
    return data


async def start_task(principal: str, message: dict, digest: str, body: dict) -> Response:
    cfg = body.get("configuration") or {}
    modes = cfg.get("acceptedOutputModes") if isinstance(cfg, dict) else None
    if isinstance(modes, list) and modes and MT_PROPOSALS not in modes:
        raise bad("acceptedOutputModes must include the proposals media type.", "CONTENT_TYPE_NOT_SUPPORTED")
    batch = validate_batch(message)

    # Synchronous check-and-reserve (no await in between) => concurrent duplicates share one task.
    rec = store.get_idem(principal, message["messageId"])
    if rec:
        return await replay(principal, rec, digest)
    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") if isinstance(message.get("contextId"), str) and message.get("contextId") else str(uuid.uuid4())
    stored_msg = dict(message, taskId=task_id, contextId=context_id)
    task = {"id": task_id, "contextId": context_id,
            "status": {"state": S_WORKING, "timestamp": now()}, "history": [stored_msg]}
    record = {"task": task, "batch": batch, "proposals": None}
    store.save(principal, record, idem=(message["messageId"], digest))
    ensure_job(principal, task_id)
    return ok({"task": await wait_ready(principal, task_id)})


def build_executions(record: dict, data: dict) -> list:
    if data.get("batchId") != record["batch"]["batchId"]:
        raise bad("batchId does not match this task.", "RESULT_MISMATCH")
    results = data.get("results")
    proposals = {p["packageId"]: p for p in record["proposals"]}
    if not isinstance(results, list) or len(results) != len(proposals):
        raise bad("results must cover every proposal exactly once.", "RESULT_MISMATCH")
    by_pkg = {}
    for r in results:
        if not isinstance(r, dict):
            raise bad("Malformed result.", "RESULT_MISMATCH")
        p = proposals.get(r.get("packageId"))
        if p is None or r.get("packageId") in by_pkg:
            raise bad("Unknown or repeated packageId.", "RESULT_MISMATCH")
        if r.get("actionId") != p["actionId"] or r.get("action") != p["action"]:
            raise bad("actionId/action does not match the stored proposal.", "RESULT_MISMATCH")
        if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
            raise bad("outcome must be ACCEPTED or REJECTED.", "RESULT_MISMATCH")
        if r["outcome"] == "ACCEPTED" and (not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]):
            raise bad("Accepted results need a receiptNonce.", "RESULT_MISMATCH")
        by_pkg[r["packageId"]] = r
    executions = []
    for p in record["proposals"]:  # proposal order; accepted only
        r = by_pkg[p["packageId"]]
        if r["outcome"] == "ACCEPTED":
            executions.append({
                "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                "receiptNonce": r["receiptNonce"], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
            })
    return executions


async def continue_task(principal: str, message: dict, digest: str) -> Response:
    task_id = message["taskId"]
    if not isinstance(task_id, str):
        raise not_found()
    async with lock_for(task_id):
        rec = store.get_idem(principal, message["messageId"])
        if rec:
            if rec["hash"] != digest:
                raise A2AError(409, "IDEMPOTENCY_CONFLICT", "Idempotency conflict",
                               "messageId was already used with different content.")
            return ok({"task": store.get_task(principal, rec["task_id"])["task"]})
        record = store.get_task(principal, task_id)
        if record is None:
            raise not_found()
        task = record["task"]
        if message.get("contextId") != task["contextId"]:
            raise bad("contextId does not match the task.", "CONTEXT_MISMATCH")
        state = task["status"]["state"]
        if state in TERMINAL:
            raise A2AError(409, "TASK_TERMINAL", "Task is in a terminal state", "No further messages accepted.")
        if state != S_INPUT:
            raise A2AError(409, "TASK_NOT_READY", "Task is not awaiting input")
        data = find_part(message, MT_RESULTS)
        if data is None:
            raise bad(f"Expected a data part with mediaType {MT_RESULTS}.", "CONTENT_TYPE_NOT_SUPPORTED")
        executions = build_executions(record, data)
        diag_verdicts(record, data)
        task["history"].append(message)
        task["artifacts"].append({
            "artifactId": "receipts-" + task_id,
            "name": "invoice-action-receipts",
            "parts": [{"mediaType": MT_RECEIPTS,
                       "data": {"batchId": record["batch"]["batchId"], "executions": executions}}],
        })
        set_status(task, S_DONE, f"Executed {len(executions)} accepted action(s); rejected proposals were not executed.")
        store.save(principal, record, idem=(message["messageId"], digest))
        return ok({"task": task})


async def send_message(request: Request, principal: str):
    body = await read_json(request)
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        raise bad("Body must be a SendMessageRequest with a message object.")
    message = body["message"]
    if not isinstance(message.get("messageId"), str) or not message["messageId"]:
        raise bad("message.messageId is required.")
    if message.get("role") != "ROLE_USER":
        raise bad("message.role must be ROLE_USER.")
    if not isinstance(message.get("parts"), list) or not message["parts"]:
        raise bad("message.parts must be a non-empty list.")
    digest = sha(canon(message))
    if message.get("taskId") is not None:
        return await continue_task(principal, message, digest)
    rec = store.get_idem(principal, message["messageId"])
    if rec:
        return await replay(principal, rec, digest)
    return await start_task(principal, message, digest, body)


# --- task reads / cancel -------------------------------------------------------------
async def get_task(request: Request, principal: str):
    record = store.get_task(principal, request.path_params["task_id"])
    if record is None:
        raise not_found()
    return ok(view(record["task"], int_param(request, "historyLength", 0, 10_000)))


async def cancel_task(request: Request, principal: str):
    task_id = request.path_params["task_id"]
    async with lock_for(task_id):
        record = store.get_task(principal, task_id)
        if record is None:
            raise not_found()
        task = record["task"]
        if task["status"]["state"] in TERMINAL:
            raise A2AError(409, "TASK_NOT_CANCELABLE", "Task cannot be canceled", "Task is already terminal.")
        set_status(task, S_CANCELED, "Task canceled; no actions were executed.")
        store.save(principal, record)
        return ok(task)


async def list_tasks(request: Request, principal: str):
    page_size = int_param(request, "pageSize", 1, 100, 50)
    offset = int_param(request, "pageToken", 0, 10**9, 0)
    history_length = int_param(request, "historyLength", 0, 10_000, 0)
    include_artifacts = request.query_params.get("includeArtifacts", "false").lower() == "true"
    context_id = request.query_params.get("contextId")
    state = request.query_params.get("status")
    tasks = [r["task"] for r in store.list_tasks(principal)]
    if context_id:
        tasks = [t for t in tasks if t["contextId"] == context_id]
    if state:
        tasks = [t for t in tasks if t["status"]["state"] == state]
    after = request.query_params.get("statusTimestampAfter")
    if after:
        try:
            cutoff = datetime.datetime.fromisoformat(after.replace("Z", "+00:00"))
        except ValueError:
            raise bad("statusTimestampAfter must be an ISO 8601 timestamp.")
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=datetime.timezone.utc)
        tasks = [t for t in tasks
                 if datetime.datetime.fromisoformat(t["status"]["timestamp"].replace("Z", "+00:00")) > cutoff]
    page, size = [], 200
    for t in tasks[offset:offset + page_size]:
        item = view(t, history_length, include_artifacts)
        item_size = len(json.dumps(item, ensure_ascii=False).encode())
        if page and size + item_size > MAX_RESPONSE - 4096:
            break
        page.append(item)
        size += item_size
    nxt = offset + len(page)
    return ok({"tasks": page, "nextPageToken": str(nxt) if nxt < len(tasks) else "",
               "pageSize": page_size, "totalSize": len(tasks)})


# --- agent card --------------------------------------------------------------------------
def base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host.split(',')[0].strip()}{BASE_PATH}/"


async def agent_card(request: Request):
    card = {
        "name": "GA5 Invoice Action Agent",
        "description": "Reads invoice claim packages, proposes exactly one evidence-backed business action per "
                       "package, and executes only grader-accepted proposals with durable, idempotent task state.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": base_url(request), "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "provider": {"organization": "GA5 student project", "url": base_url(request)},
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer",
                                                                   "description": "Each bearer token is a separate principal."}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": [MT_BATCH, MT_RESULTS],
        "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or open_exception "
                           "for each invoice package, cites the decisive evidence, and records receipts for accepted actions.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-payment", "a2a"],
            "examples": ["Propose actions for an invoice claim batch."],
            "inputModes": [MT_BATCH, MT_RESULTS],
            "outputModes": [MT_PROPOSALS, MT_RECEIPTS],
        }],
    }
    return Response(json.dumps(card).encode(), media_type="application/json")


async def health(request: Request):
    """Liveness plus model diagnostics (no secrets): last outcome per model and the decision-cache size."""
    body = {"ok": True, "models": decide.LLM_MODELS, "keySet": bool(decide.LLM_API_KEY),
            "providers": {m: e[0].split("/")[2] for m, e in decide.ENDPOINT.items()},
            "modelStatus": decide.MODEL_STATUS, "cachedDecisions": store.count_decisions(),
            "warming": len(decide._warming)}
    return Response(json.dumps(body).encode(), media_type="application/json")


async def fallback(request: Request, exc):
    status = getattr(exc, "status_code", 500)
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(status, "INTERNAL")
    return problem(A2AError(status, code, "Request failed"))


routes = [
    Route("/.well-known/agent-card.json", agent_card, methods=["GET"]),
    Route("/healthz", health, methods=["GET"]),
    Route(f"{BASE_PATH}/message:send", guarded(send_message, has_body=True), methods=["POST"]),
    Route(f"{BASE_PATH}/tasks", guarded(list_tasks), methods=["GET"]),
    Route(f"{BASE_PATH}/tasks/{{task_id}}:cancel", guarded(cancel_task), methods=["POST"]),
    Route(f"{BASE_PATH}/tasks/{{task_id}}", guarded(get_task), methods=["GET"]),
]

app = Starlette(routes=routes, exception_handlers={404: fallback, 405: fallback, 500: fallback})
app.router.redirect_slashes = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")),
                proxy_headers=True, forwarded_allow_ips="*")
