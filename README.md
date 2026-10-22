# A2A Invoice Action Agent (GA5)

A2A 1.0 HTTP+JSON agent that proposes one evidence-backed action per invoice package, waits for grader results,
and executes only ACCEPTED proposals. It keeps durable, idempotent task state in SQLite.

| File | Layer |
|---|---|
| `app.py` | Protocol: Agent Card, auth, version and media-type checks, lifecycle, idempotency, isolation, cancel race |
| `store.py` | SQLite storage. Every change is committed before the response is sent. |
| `decide.py` | AI decisions: batched model calls, evidence snapping, amount conversion, cache by package content |
| `test_contract.py` | Local self-test using synthetic data. It uses none of your grader checks. |

## 1. Configure a model (any OpenAI-compatible endpoint)

| Provider | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| Google Gemini (free tier) | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-3.6-flash,gemini-3.5-flash,gemini-3.8-flash` |
| AI Pipe (IITM) | `https://aipipe.org/openai/v1` | `gpt-4.1-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | any model |
| Ollama (local) | `http://localhost:11434/v1` | e.g. `qwen2.5:14b` |

`LLM_MODEL` can be a comma-separated list. When a model answers 404 (retired), 429 (quota used up) or 5xx
(overloaded), the agent tries the next model. Free-tier Gemini quotas are per model and small (for example 20
requests per day), so list several. Only if every model fails does the agent fall back to a keyword heuristic;
fallback decisions are much less accurate, are never cached, and log `[decide] FALLBACK used for package ...`.

## 2. Run locally and expose over public HTTPS (Cloudflare quick tunnel)

```powershell
$env:LLM_API_KEY="..."; $env:LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"; $env:LLM_MODEL="gemini-3.6-flash,gemini-3.5-flash,gemini-3.8-flash"; $env:LLM_REASONING_EFFORT="low"; $env:PORT="8765"; python app.py
```

In a second terminal (install once with `winget install Cloudflare.cloudflared`):

```powershell
cloudflared tunnel --url http://localhost:8765
```

Submit `https://<random>.trycloudflare.com/a2a/` exactly, including the trailing slash. Keep both windows open
while the grader runs. The tunnel URL changes every time you restart the tunnel.

## 3. Or deploy to a Hugging Face Space (Docker, free, always-on HTTPS)

Create a Docker Space and upload `Dockerfile`, `requirements.txt`, `app.py`, `store.py` and `decide.py`. Add
`LLM_API_KEY` as a Secret, and add `LLM_BASE_URL`, `LLM_MODEL` and `PUBLIC_BASE_URL=https://<user>-<space>.hf.space/a2a/`
as Variables. Then submit that same URL.

## 4. Before spending a grader check

```powershell
python test_contract.py https://<your-public-host>/a2a/
```

Every line should say PASS. The self-test uses no grader checks. `python test_decide.py` runs offline checks of the
evidence and rationale validation.

## Knobs

* `EVIDENCE_REF_BRACKETS=true` returns refs as `[REF]` instead of `REF`. Flip it only if the grader feedback reports evidence-format failures.
* `LLM_FALLBACK_MODELS` is an optional extra comma-separated list appended to `LLM_MODEL`.
* `LLM_CHUNK` sets how many packages go into one model call (default 6, so a 12-package core makes 2 parallel calls).
* `REQUEST_BUDGET_S` sets the per-request time budget (default 40 s; the grader allows 45 s).
* `ALLOWED_TOKENS` is an optional comma-separated allowlist. By default every bearer token is a separate principal.
