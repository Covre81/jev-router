# jev-router

A local, Anthropic-compatible gateway for Claude Code (or any Messages-API
client) that uses **TypeSafe JEV** as a System One classifier to keep simple
work off your Claude quota.

- **Simple work** goes to a cheap OpenAI-compatible provider: Ollama Cloud
  (default, free plan), Groq or OpenRouter.
- **Structural work** goes to Anthropic.

```
                                      ┌─ simple ───▶ Ollama Cloud / Groq / OpenRouter  (/chat/completions, translated)
Claude Code ──▶ jev-router :8787 ──┤
                    │                 └─ structural ─▶ Anthropic          (/v1/messages, byte-for-byte)
                    └─ new human turn ─▶ JEV /v1/systemone
```

## Routing model (ADR) — the adopted design

**Status:** adopted. It replaces the earlier Haiku → Sonnet → Opus ladder.

**Context.** On a Claude subscription, routing *between Claude models* cuts no
bill. Every model draws from the same quota. On top of that, switching models
mid-conversation discards the model-scoped prompt cache. Moving between Claude
models only moves the cost around. The goal is to **spend the Claude quota
only on work that needs Claude**.

**Decision.** Use two providers with different billing:

| Route | Provider | Billing | Gets |
|---|---|---|---|
| `cheap`, trivial tier | Any OpenAI-compatible API (default: Ollama Cloud `gpt-oss:20b-cloud`, `CHEAP_MODEL`) | Free plan with usage limits, or per token on Groq/OpenRouter | JEV level 1, the simple tasks: questions, explanations, one-file edits, renames, docstrings |
| `cheap`, standard tier | Same endpoint, bigger model (default `gemma4:31b-cloud`, `CHEAP_MODEL_STANDARD`; `off` disables it) | Same | Level 2, standard feature work, when it carries no risk |
| `primary` | Anthropic, with whatever model the client picked | Your Claude quota or API key | Level 3, structural work: Clean Architecture refactors, heavy test design, concurrency, security and performance. Plus everything the cheap route can't carry |

**Rules.**

1. **System One decides the route.** JEV answers one ordinal `score`
   question, and the router gets the probability distribution
   `{simple, standard, structural}`.
2. **The cheap route needs confidence, not just a majority.** A task goes
   to the trivial tier only when `P(simple) ≥ ROUTER_MIN_CHEAP_PROBABILITY`
   (default 0.9). It does not use the most likely level: `{.45, .30, .25}`
   has "simple" as its top level, but a 55% chance the small model is out of
   its depth. That turn goes to the standard tier instead when
   `P(simple) + P(standard) ≥ ROUTER_MIN_STANDARD_PROBABILITY` (default
   0.75), and to Anthropic otherwise. With `CHEAP_MODEL_STANDARD=off`, level
   2 stays primary unless the legacy `ROUTER_STANDARD_ROUTE=cheap`.
3. **The tier sticks for the whole conversation, and only moves up**
   (trivial → standard → primary).
   - Tool-result turns of the agent loop reuse the tier without calling JEV.
   - A new human turn on a cheap tier is re-classified and may escalate
     (`escalated:standard`, `escalated`).
   - Primary is terminal: going back down would throw away the Anthropic
     cache, and the next escalation would pay to rebuild it. The one
     exception is critical quota (rule 5).
   - Compaction does not reopen the cheap route: a session on Claude stays
     there after `/compact` (`pinned:compaction`).
   - Each tier has its own context budget (`CHEAP_CONTEXT_TOKENS`,
     `CHEAP_STANDARD_CONTEXT_TOKENS`); past it, the turn goes primary.
4. **Failure only ever moves work up to Anthropic.**
   - The request can't be translated (images, documents, server tools): primary.
   - The request is larger than the cheap model's context budget: primary.
   - The cheap provider returns an error or times out: it is retried once quickly (`CHEAP_RETRY=true`), then falls back to primary and the conversation is pinned there.
   - The cheap provider breaks mid-stream: an Anthropic `error` event is sent,
     the conversation is pinned to primary, and Claude Code retries (2.1.289
     re-sends the turn as a non-streaming request). The pin does not cover a
     break on a conversation's first turn: that retry is re-classified.
   - JEV fails or times out: primary.
5. **The Claude quota opens the cheap route when it runs low**
   (`QUOTA_ROUTING=true`). Every Anthropic response carries the quota
   (`anthropic-ratelimit-unified-5h-*`, `-7d-*`, …); the router keeps the
   window closest to its limit.
   - From `QUOTA_PRESSURE` (80%) on, the cheap bars drop to
     `QUOTA_PRESSURE_MIN_CHEAP` (0.7) and `QUOTA_PRESSURE_MIN_STANDARD` (0.6)
     for new turns and subagents (`quota:pressure`). Auxiliary traffic stays
     on Claude.
   - From `QUOTA_CRITICAL` (95%) on, a new human turn of a session pinned to
     Claude is re-classified and may go cheap (`quota:reclassified`); it then
     stays there, because going back would rebuild the Claude cache. The
     next structural turn escalates as usual.
   - When Anthropic answers 429 or 529, the request is retried on the cheap
     provider (`FAILOVER_ON_PRIMARY_RATE_LIMIT=true`, now the default): a
     degraded answer instead of a dead session. Session-start quota probes
     (max_tokens <= 1) are never rerouted: their 429/529 reaches Claude Code unchanged,
     with its anthropic-ratelimit-* headers, so Claude Code can show its own quota warning.
   - The risk veto (rule 7) is checked before any of this, at every level.
6. **Some Claude Code traffic always stays primary**
   (`ROUTER_PRIMARY_CLASSES=auxiliary,compaction`). Claude Code's
   `auxiliary` class includes the auto-mode safety classifier, and compaction
   summaries need the strong model.
7. **Risk vetoes the cheap route, always.** The same JEV call asks three
   yes/no Nouls; if any is ≥ 0.5 the turn stays on Anthropic, however simple
   it looks:
   - `security_sensitive`: auth, password hashing, tokens, crypto, secrets;
   - `destructive_or_production`: deletes data, migrations, production;
   - `requires_inspection`: a verdict or report on the project's state
     ("is everything ok here?", "summarize what changed") that needs the
     repository investigated first. The 20B answered both of those without
     looking. Edits, named commands and explaining pasted code are excluded.
   A response missing any of the three fails toward Anthropic.
   `jev-router stats` counts cheap answers that called none of the tools
   they were offered (**inspection miss**): the misroutes the veto missed.

**Consequences.**

- **Your code goes to a third party.** The cheap provider sees the full prompt
  and the file contents the agent reads. Check its data policy before pointing
  a client repository at it.
- **This setup is unsupported by Anthropic.** Claude Code is built for Claude
  models. Anthropic's gateway docs state that it doesn't support routing Claude
  Code to non-Claude models. On the cheap route, adaptive thinking, prompt
  caching and server tools (web search, tool search) are gone.
- **Claude Code's agent loop is demanding.** The cheap model has to drive it
  with 20+ tools and a system prompt of tens of thousands of tokens. A 20B
  model does fine on the narrow level-1 band and fails outside it. Hence the
  0.8 bar and level 2 on primary.
- **Rate limits on cheap tiers.** One Claude Code request is roughly 20–40K
  input tokens. Free and dev tiers on cheap providers often have per-minute
  token caps below that. Each rejection fails over to Claude, so you spend the
  very quota you meant to save. Check your tier's limits.
- **Claude Code shows the wrong model name on the cheap route.** It keeps
  showing the model you picked. The truth is in the `x-jev-route` response
  header and the `route decision` log line.

### Read-first hint (`CHEAP_READ_FIRST_HINT`)

On the cheap route, the model might try to guess the project's state instead of reading its files.
When `CHEAP_READ_FIRST_HINT=true` (the default), if a session's first turn is a broad question about what to do next (e.g. "what next?", "onde paramos?"), the router appends a system-reminder block at the end of the last user message instructing the model to use its file tools (Glob/LS, Read) on project files (README, CLAUDE.md, plan documents) rather than guessing or inventing details.
- **When it fires:** only on a fresh conversation's first turn, when the user's prompt matches an open-ended "what next" pattern, and file-reading tools are available. The hint is never sent to Claude or Gemini.
- **Placement:** The reminder block is appended to the last user message, wrapped in `<system-reminder>` tags. (Note: the `gpt-oss` model receives reasoning in a separate field which the router drops, so any reasoning-like text the client sees was generated directly in the answer by the model).
- **Standard Tier Promotion (`CHEAP_READ_FIRST_STANDARD`):** By default (`true`), these read-first turns are promoted to the standard tier model (e.g. `gemma4:31b-cloud`), avoiding the trivial tier. If the standard tier fails, or is disabled (`off`), it falls back to the trivial tier model.
- **Follow-up reminder:** On the trivial and standard tiers, if the model replies with a tool call (like LS) instead of reading a file, the next tool-result turn carries a follow-up reminder to read the files. This fires until a file has been read, for at most 6 assistant turns, and is never sent to Claude or Gemini.
- **How to turn it off:** set `CHEAP_READ_FIRST_HINT=false` in your `.env`.

## Gemini tier (Antigravity CLI subscription)

The router can optionally divert medium-to-hard, text-only turns to Gemini using the user's Google subscription via the Antigravity CLI (`agy`) as a child process. This spares the Claude quota for work that needs it, without requiring a separate API key.

- **Scope**: Text-only turns (explanations, design discussions, planning, conceptual Q&A, code reviews) that do not require tool use, evaluated after `gpt-oss:20b-cloud` (trivial) and `gemma4:31b-cloud` (standard) and before Claude (primary).
- **Fail-open**: It is off by default and fails open to the session's fallback tier or Claude (e.g. if the CLI is missing, times out, or the tier is busy).
- **Latency**: ~6–7 s minimum per turn. The answer arrives all at once (simulated streaming). Long explanations with `gemini-3.8-flash-high` can take ~15–25 s, so `GEMINI_TIER_TIMEOUT_MS` may need 90000 for long answers.
- **API Keys**: The child process is started WITHOUT `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT` (stripped from its environment), so agy always uses the Google account login (subscription) and never an API key.
- **JEV Question**: Uses the `text_answer_suffices` question to determine if the turn can be fully answered with a written reply without running tools.
- **Isolation and Safety**: The child process runs with a router-owned `agy` profile (`HOME`/`USERPROFILE`) containing explicit deny rules for all tools (`write_file(*)`, `command(*)`, etc.). The temporary working directory is removed immediately. If the model attempts any tool use, the child process is killed instantly and the turn falls back to Claude.

**Configuration (Environment variables):**
- `GEMINI_TIER=on` (default: `off`)
- `GEMINI_TIER_MODEL=gemini-3.8-flash-high`
- `GEMINI_TIER_BIN` (default: `%LOCALAPPDATA%\agy\bin\agy.exe` on Windows, `agy` elsewhere)
- `GEMINI_TIER_TIMEOUT_MS=60000`
- `GEMINI_TIER_MIN_TEXT_ONLY=0.8` (min JEV confidence that a text answer suffices)
- `GEMINI_TIER_PRESSURE_MIN_TEXT_ONLY=0.6` (bar when Claude quota is pressured)
- `GEMINI_TIER_FROM_PRIMARY=true` (evaluate text-only human turns even for Claude-pinned sessions. Note: this costs one extra JEV call per human turn of a Claude session)
- `GEMINI_TIER_MAX_CONCURRENCY=1` (Queueing is skipped: if busy, it fails over to the next tier)
- `GEMINI_TIER_MAX_PROMPT_CHARS=200000` (Max rendered transcript characters. Older messages are omitted if exceeded)
- `GEMINI_TIER_BREAKER_FAILURES=3` (Consecutive failures before opening the circuit breaker)
- `GEMINI_TIER_BREAKER_COOLDOWN_MS=300000` (Cooldown ms before a half-open trial)
- `GEMINI_TIER_HOME` (Custom HOME/USERPROFILE for the isolated agy profile. Defaults to `<jevHome()>/agy-home`)

**How to enable:**
1. Ensure `agy` is installed and logged in once interactively.
2. Set `GEMINI_TIER=on` in your `.env`.
3. Run `npm run build`.
4. Run `jev-router reload` (or restart the task).
5. Check `/healthz` to verify it shows `gemini`.
6. Watch `jev-router stats`.

## Repository structure

```
src/
├── cli.ts                         # `jev-router` binary: serve | reload | stats | statusline | context | agent-hook | help; env-file loading
├── serve.ts                       # composition root (only file that knows concrete classes)
├── supervisor.ts                  # reverse proxy over a forked worker: reload without dropping sessions
├── build-info.ts                  # dist/build-info.json (sha + build time) for /healthz and the status line
├── quota.ts                       # anthropic-ratelimit-* headers → binding quota window and level
├── statusline.ts                  # `jev-router statusline`: health probe + session's last route
├── agents.ts                      # `jev-router agent-hook`: running-subagent registry under ~/.jev-router/agents
├── hook-input.ts                  # reads the JSON Claude Code pipes to a status line or hook
├── config.ts                      # zod-validated env → typed Config; fails fast at boot
├── paths.ts                       # ~/.jev-router (or $JEV_ROUTER_HOME): .env + telemetry.db
├── domain/                        # pure, no I/O
│   ├── complexity.ts              # ComplexityDistribution (JEV levels 1/2/3)
│   └── policy.ts                  # selectRoute(), stickyRoute(): deterministic decisions
├── classifier/                    # System One port + adapters
│   ├── classifier.ts              # ComplexityClassifier interface
│   ├── jev-classifier.ts          # TypeSafe /v1/systemone adapter
│   └── heuristic-classifier.ts    # offline mock for dev/tests
├── routing/                       # decides; never talks to a provider
│   ├── messages-body.ts           # open-schema parsing of the Anthropic body
│   ├── router.ts                  # stickiness, escalation, context budget, fail-to-primary
│   └── session-store.ts           # TTL + LRU map
├── providers/                     # executes; every provider answers in Anthropic format
│   ├── provider.ts                # Provider port: response | unavailable
│   ├── anthropic.ts               # byte-level forwarder (primary)
│   ├── cheap-health.ts            # polls the cheap provider's /models: skip it while it is down
│   └── openai/
│       ├── provider.ts            # OpenAI-compatible provider (cheap)
│       ├── translate-request.ts   # Anthropic Messages → Chat Completions
│       ├── translate-response.ts  # Chat Completions JSON/SSE → Anthropic Messages
│       └── sse.ts                 # SSE reader/writer
├── proxy/
│   ├── server.ts                  # Fastify routes, failover in both directions
│   └── headers.ts                 # hop-by-hop filtering, credential extraction
└── telemetry/                     # cost audit; can never fail or slow a response
    ├── schema.ts                  # `router_logs` row type, SQL migrations, insert
    ├── db.ts                      # node:sqlite + WAL, applies migrations on boot (user_version)
    ├── usage-meter.ts             # taps the relayed body (SSE/JSON, gzip/br) for `usage`
    ├── audit.ts                   # one row per exchange, recorded after the socket closes
    ├── recorder.ts                # TelemetrySink port; queued, batched SQLite writes
    ├── stats.ts                   # aggregates + table for `jev-router stats`
    └── statusline.ts              # last route per session, today's cheap share, the rendered line
```

The router returns a decision (`cheap` | `primary`) and knows nothing about
base URLs, headers or wire formats. Providers own all three. That split is
deliberate: adding a provider, or changing how Groq is called, never touches
routing rules, and routing rules are tested without HTTP.

### Translation: shallow, but never lossy

The translator maps exactly what has a 1:1 equivalent, in both directions.

**Request: Anthropic → OpenAI.**

| Anthropic | OpenAI |
|---|---|
| `system` | `system` message |
| text | text |
| `tool_use` | `tool_calls` |
| `tool_result` | `role: tool` messages, placed right after the call |
| `tools[].input_schema` | `functions[].parameters` |
| `tool_choice` | `tool_choice` |
| `stop_sequences` | `stop` |

These are dropped on purpose:

- `thinking` blocks: they are bound to the model that produced them.
- `cache_control`.
- Server tools and deferred tools: the cheap model never sees them.

Anything else throws `NotTranslatableError`, and the request goes to Anthropic.
That covers images, documents, `tool_reference` and server tool results. The
cheap model never sees a conversation with pieces silently missing.

**Response: OpenAI → Anthropic.** This direction is mandatory, because Claude
Code only parses Anthropic events.

- Text deltas stream live.
- Tool calls are buffered and emitted as complete `tool_use` blocks. Buffering
  guarantees a well-formed event sequence, and lets the proxy validate the
  argument JSON *before* Claude Code can run the tool.
- Tool call ids are sanitized to Anthropic's `^[a-zA-Z0-9_-]+$`, so a
  conversation that later escalates replays cleanly on Claude.

## Dependencies

| Package | Why |
|---|---|
| `fastify` | Raw-buffer bodies with a hard `bodyLimit`, streamed replies, pino logging |
| `undici` | One HTTP client for all three upstreams. `request()` exposes the raw, non-decompressed stream that the Anthropic passthrough needs |
| `zod` | Validates env, the body fields we read, and every JEV/OpenAI payload |

Deliberately absent:

- `dotenv`: `process.loadEnvFile` does the job.
- The Anthropic SDK: a gateway forwards bytes, it doesn't build requests.
- The OpenAI SDK: two endpoints and a stream translator don't justify it.
- An LRU library.
- An ORM: one table, a handful of queries.
- A SQLite driver package: the cost audit uses Node's built-in `node:sqlite`
  (one table, plain SQL, migrations as an append-only list applied on boot),
  so there is no native module to compile. Node 22 and 24 still flag it
  experimental and print one `ExperimentalWarning` at startup.

## Setup

```bash
git clone https://github.com/Covre81/jev-router.git && cd jev-router
npm ci
cp .env.example .env
#   TYPESAFE_API_KEY=...   (or CLASSIFIER=heuristic for offline dev)
#   Ollama Cloud (default): `ollama signin && ollama pull gpt-oss:20b-cloud`, no key
#   Groq / OpenRouter: set CHEAP_BASE_URL, CHEAP_MODEL and CHEAP_API_KEY (see presets)
npm test                   # fake upstreams, no network
npm run build && npm start
curl -s localhost:8787/healthz
```

Requires Node ≥ 22.19.

### Global CLI (`npm link`)

```bash
cd jev-router
npm ci                     # also builds dist/ (prepare script)
npm link                   # puts `jev-router` on your PATH
mkdir -p ~/.jev-router && cp .env.example ~/.jev-router/.env   # then fill in the keys
jev-router serve           # start the gateway (default command)
jev-router stats           # cost audit, from any directory
jev-router stats --since 24h
jev-router stats --json
```

Env lookup order: variables already set in the shell, then `--env <file>`,
then `./.env`, then `~/.jev-router/.env`. `JEV_ROUTER_HOME` moves the
`~/.jev-router` directory. After `git pull`, run `npm run build` again; the
link points at this checkout, so nothing else needs reinstalling.

### Reload without dropping sessions

Every Claude Code session goes through the router, so restarting it used to
cut them all. `jev-router serve` now runs as a **supervisor**: it owns
`HOST:PORT`, forks the real gateway as a worker on an ephemeral loopback port,
and pipes every request to it (unbuffered, so SSE streams flow as written).

```bash
npm run build              # runs `reload` itself via the `postbuild` hook
jev-router reload          # manual, e.g. after editing .env: node dist/cli.js reload --env .env
```

`postbuild` ignores a missing supervisor (router not running, or `npm install`
triggering `prepare`), so a build never fails because of it.

`reload` asks the supervisor (loopback-only control port, `CONTROL_PORT`,
default `PORT + 1`) to start a fresh worker from the new `dist/`. Once it
listens, new requests go to it; the old worker finishes its open streams and
is cut after `RELOAD_DRAIN_MS` (default 2 min). The worker re-reads the env
file, so `.env` changes also apply on `reload`; only `HOST`, `PORT`,
`CONTROL_PORT` and `RELOAD_DRAIN_MS` belong to the supervisor and need a
restart. A crashed worker is restarted after 1, 2, 4, 4 and 4 s; after that
the supervisor answers 503 and waits for a `reload`. A change to
`supervisor.ts` itself also needs a restart.

`/healthz` reports which build answers:
`{ok, sha, builtAt, startedAt, pid, cheap, quota}`, where `quota` is the
last Claude quota reading (`{utilization, window, level, …}`, or null before
the first Anthropic response) and `cheap` is the cheap
provider's health (`unknown | up | down`). The build stamps
`dist/build-info.json` (`git rev-parse --short=12 HEAD` + time). For
`npm run dev` (tsx watch), set `SUPERVISOR_ENABLED=false`.

**Cheap provider health.** The worker polls `GET {CHEAP_BASE_URL}/models`
every `CHEAP_HEALTH_INTERVAL_MS` (30 s). While it is `down`, a turn routed
cheap goes to Anthropic with reason `skipped:cheap-unhealthy`, without
spending an attempt and without pinning the conversation to Anthropic: it
returns to cheap once the provider is back. Caveat: Ollama's `/models` proves
the daemon is up, not that ollama.com serves a `*-cloud` model.

**Cheap provider presets:**

```bash
# Ollama Cloud via the local daemon (default, free plan)
CHEAP_BASE_URL=http://127.0.0.1:11434/v1
CHEAP_API_KEY=ollama
CHEAP_MODEL=gpt-oss:20b-cloud

# Groq
CHEAP_BASE_URL=https://api.groq.com/openai/v1
CHEAP_MODEL=openai/gpt-oss-20b

# OpenRouter
CHEAP_BASE_URL=https://openrouter.ai/api/v1
CHEAP_MODEL=<any tool-calling model id from openrouter.ai/models>
```

Ollama Cloud runs the model on ollama.com: the prompt, including your code,
leaves the machine, and the free plan has usage limits. When a limit is hit
the request fails before the first byte and the router falls back to
Anthropic. A local model (`qwen2.5:14b`, `llama3.1`) keeps everything on the
machine but is much slower on CPU; run `bench.ts` before choosing it.

On Groq, `llama-3.1-8b-instant` was deprecated for free and dev tiers on
2026-08-16, with `openai/gpt-oss-20b` as the recommended replacement. That is
why the default is not "Llama 3 8B".

**Claude credentials:** see `UPSTREAM_AUTH_MODE` in `.env.example`.

- `passthrough` (default): the proxy holds no Anthropic secret and forwards
  whatever Claude Code sends, API key or claude.ai login.
- `inject`: the proxy holds `ANTHROPIC_API_KEY` and replaces client
  credentials. Off-loopback, `PROXY_AUTH_TOKEN` is required.

The cheap provider's key never leaves the proxy, and Anthropic credentials
are never sent to it.

## Claude Code integration

`~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1"
  }
}
```

- **`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` is required for rule 6.** Without it,
  Claude Code doesn't send `x-claude-code-request-class`, and its side
  requests (titles, classifiers, summaries) get classified like any prompt.
- **With a claude.ai login,** the primary route keeps using your
  subscription. Only cheap-routed turns leave it.
- **Other clients:** point the base URL at `http://127.0.0.1:8787`. Without
  session headers, conversations are keyed by a hash of their first message.

**Status line.** To see under every Claude Code prompt whether the router is
up and where the session's last turn went, add to the same `settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "node /path/to/jev-router/dist/cli.js statusline --env /path/to/jev-router/.env"
  }
}
```

It prints `jev-router ✓ · last: cheap (JEV 0.93) · today 12/40 cheap`, or
`jev-router ✗ offline` when `/healthz` doesn't answer within 500 ms.
`⚠ build velho` means `dist/build-info.json` is newer than the build that
answers `/healthz` (built but not reloaded: run `jev-router reload`);
`cheap ✗` means the health check sees the cheap provider down.
With `STATUSLINE_QUOTA=1` in the env file it also appends `cota 82% 5h`,
the binding Claude quota window as the router last saw it
(`⚠` from 80%, where the cheap bars drop); readings older than 6 h are not
shown. It is off by default: Claude Code's own status line JSON already
carries the plan's 5h/7d usage. The
reason in parentheses is JEV's P(simple), or the route reason when JEV wasn't
asked (`sticky`, `auxiliary`, …). `today 12/40 cheap` counts today's routed
turns, not requests: main-agent human turns plus every turn JEV scored
(subagents included). Auxiliary requests and tool-call continuations are not
routing decisions. Without the hint headers only JEV-scored turns count.
`ctx 125k` is the context the session's main agent sent on its last request
(input + cache); from 200k on it turns into `⚠ ctx 412k → /compact or /clear`,
because every tool call re-reads it all — in the week to 2026-10-06, requests
above 200k carried 92% of main-session tokens. Pass `--env` explicitly: the
status line runs in each project's directory, and that project's own `.env`
could set another `PORT`. Adjust the paths to your checkout (or use
`jev-router statusline` after `npm link`).

`🤖 2 ⏳ 1h05` counts the session's running subagents (Agent tool) and how
long the oldest has run. It needs `jev-router agent-hook` on four hook events
in the same `settings.json`; without them it never shows:

```json
{
  "hooks": {
    "SessionStart":  [{ "matcher": "", "hooks": [{ "type": "command", "command": "node /path/to/jev-router/dist/cli.js agent-hook --env /path/to/jev-router/.env", "timeout": 5 }] }],
    "SubagentStart": [{ "matcher": "", "hooks": [{ "type": "command", "command": "node /path/to/jev-router/dist/cli.js agent-hook --env /path/to/jev-router/.env", "timeout": 5 }] }],
    "SubagentStop":  [{ "matcher": "", "hooks": [{ "type": "command", "command": "node /path/to/jev-router/dist/cli.js agent-hook --env /path/to/jev-router/.env", "timeout": 5 }] }],
    "SessionEnd":    [{ "matcher": "", "hooks": [{ "type": "command", "command": "node /path/to/jev-router/dist/cli.js agent-hook --env /path/to/jev-router/.env", "timeout": 5 }] }]
  }
}
```

SubagentStart writes an empty file `~/.jev-router/agents/<session>/<agent>`
(its mtime is the start), SubagentStop deletes it, SessionEnd deletes the
session's folder. A Claude Code that crashes fires neither, so SessionStart
cleans up: on `resume` it drops the session's agents (they died with the old
process), and on any start it deletes session folders untouched for 7 days.
The hook always exits 0, so it can never block a subagent from stopping.

A status line that wraps this one and colours it can pass `--json` instead of
parsing the markers: it prints `{"text":"jev-router ✓ · …","level":"ok"}`,
where `level` is `critical` when the router is offline, `warn` when the line
flags anything (`⚠`, `cheap ✗`, a last turn that did not end ok), and `ok`
otherwise.

## Telemetry and cost audit

Every `/v1/messages` exchange is written to `~/.jev-router/telemetry.db`
(table `router_logs`, override with `TELEMETRY_DB_PATH`, disable with
`TELEMETRY_ENABLED=false`).

| Column | Meaning |
|---|---|
| `session_id` | Conversation key: `<x-claude-code-session-id>:<agent>`, or a hash of the first message |
| `human_prompt_hash` | sha256 of the human-typed text; null on tool-result turns. Equal hashes = repeated prompt |
| `jev_decision` | JSON `{simple, standard, structural, pSimple, pComplex, classifierMs, tokensIn, tokensOut, risk, riskScores, textOnly, model}`; `pComplex` is level 3, `tokensIn/Out` are the classifier's own usage, `riskScores` holds each risk Noul (`risk` is their max), `textOnly` is null unless the Gemini tier is on. Null when JEV was not asked (sticky turns, auxiliary requests) |
| `final_provider` | `anthropic` or `openai` (the cheap OpenAI-compatible provider) |
| `tokens_in` / `tokens_out` | From the response `usage`; `tokens_in` includes cache reads and writes |
| `cache_read_tokens` | Part of `tokens_in` served from the Anthropic prompt cache |
| `latency_ms` | Request received → last byte sent to the client |
| `fallback_triggered` | The cheap provider failed before answering and Anthropic served the request |
| `outcome` | `ok`, `http_error`, `stream_error`, `client_abort` or `proxy_error` |
| `upstream_status`, `upstream_ratelimit_headers`, `upstream_failure` | Status, Anthropic rate-limit header names, and failure reason from the upstream provider |

**Why it adds no latency.** The response body passes through a tap that
copies nothing and only scans `data:` lines that mention `usage` or `error`.
Compressed bodies are decoded on a side channel. The row is built after the
client socket closes, then pushed to an in-memory queue. A timer writes the
queue to SQLite once a second, in one transaction. If a write fails, the rows
stay queued and the response is unaffected.

**`jev-router stats`** prints the total requests, how many were diverted from
Anthropic, fallbacks, repeated prompts, tokens per provider and the
**net cost in dollars**, which says whether routing is making or losing money.

The net cost is measured against a **baseline**: the same traffic sent to
Anthropic with a warm prompt cache. For each turn, the history the previous
turn already sent is billed as a cache read, and only the new tokens are
billed as a cache write.

| Line | How it is computed |
|---|---|
| Gross savings | For each request the cheap provider served: warm Anthropic cost minus cheap cost |
| Cache-miss penalty | First Anthropic request after a cheap one (escalation or fallback): actual cost minus warm cost. Anthropic re-reads the whole history at full input + write price because the cheap turns broke the cache |
| Failed cheap attempts | Tokens billed by the cheap provider on requests that then fell back |
| Classifier (JEV) calls | Tokens each JEV classification billed (from its `usage`), at `JEV_PRICE_*`: the router's own overhead |
| **NET** | baseline − (actual Anthropic + actual cheap + JEV). Positive = **PROFIT**, negative = **LOSS** |

Prices: Anthropic list prices per model (built-in table, override with
`PRIMARY_PRICE_*`). The cheap provider price defaults to $0, which is right
only for a local model; set `CHEAP_PRICE_INPUT_PER_MTOK` and
`CHEAP_PRICE_OUTPUT_PER_MTOK` for Groq or OpenRouter, otherwise `stats` warns.
TypeSafe bills JEV per token: set `JEV_PRICE_INPUT_PER_MTOK` and
`JEV_PRICE_OUTPUT_PER_MTOK` from your plan, otherwise `stats` warns once JEV has
scored a request. A failed JEV call reports no usage and is not counted.
On a Claude subscription the dollars are API-equivalent quota, not a bill.

Known limits:

- the cheap model's tokenizer is not Claude's, so its token counts are only
  close to what Anthropic would have counted;
- rows logged before the `cache_write_tokens` column existed are priced as if
  they had no cache writes;
- served (`ok`) requests whose response carried no `usage` are counted and
  reported, but priced at $0; failed requests never carry usage and are not
  counted.


### Diagnostics (R0)

When `TELEMETRY_DIAGNOSTICS=true` (default: false) is set, the router logs additional metadata into the telemetry database for deep inspection of agent behavior, context management, and rate limits.
To respect privacy, only hashes (e.g. `system_hash`, `tools_hash`), lengths (`system_chars`), token counts, and header names are stored; full strings, secrets or prompt contents are never recorded.

Additional `jev-router stats` flags are available for diagnostic queries:
- `--by-class`: aggregates usage and proxy USD costs split by `x-claude-code-request-class` and `requested_model`.
- `--cache-misses [--min-write N]`: lists every Anthropic turn that incurred a cache write penalty larger than N (default 150000). It breaks down each cache-miss cause (e.g., `first-turn-of-session`, `return-from-cheap`, `failover-from-cheap`, `compaction`, `model-switch`, `system-changed`, `tools-changed`, `gap>1h`, `gap5-60m`) and provides a summary.
- `--daily`: aggregates traffic and USD costs per day, and prints overall task count, median/p90 USD per task, and the top 10 most expensive tasks.

Before tuning `ROUTER_MIN_CHEAP_PROBABILITY`, use the database to measure:

- the share of turns that went cheap;
- how many of those escalated on the next human turn (a misroute signal);
- how often the cheap provider failed over.

```sql
-- conversations that used both providers (escalated or failed over)
SELECT session_id FROM router_logs GROUP BY session_id
HAVING min(final_provider) = 'anthropic' AND max(final_provider) = 'openai';
```

The cheap route can also fail silently: the model says it is done and the work
is wrong. The user's tell is sending the same prompt again. Compare that rate
per provider; if `openai` sits well above `anthropic`, raise the bar further or
try another model with `bench.ts`:

```sql
-- Prompts repeated after an answer, by the provider that served the first send.
WITH firsts AS (
  SELECT human_prompt_hash, final_provider,
         count(*) OVER (PARTITION BY human_prompt_hash) AS sends,
         row_number() OVER (PARTITION BY human_prompt_hash ORDER BY id) AS nth
  FROM router_logs WHERE human_prompt_hash IS NOT NULL
)
SELECT final_provider, count(*) AS prompts, sum(sends > 1) AS repeated,
       round(100.0 * sum(sends > 1) / count(*), 1) AS repeated_pct
FROM firsts WHERE nth = 1 GROUP BY final_provider;
```

## Checks against real APIs (local only)

CI runs only the offline suite (`npm test`: every upstream is a local fake).
Three scripts call real services and are meant to be run by hand. All of them read the
same env files as the CLI (`--env <file>`, else `./.env`, else
`~/.jev-router/.env`).

**JEV contract**: one `POST /v1/systemone` with a minimal prompt. Prints the
raw body, then checks it strictly against the adapter's schema (levels
`0`/`1`/`2` present, summing to 1). Exit 0 = contract holds.

```bash
npx tsx scripts/test-jev-real.ts
```

**Cheap-model tool benchmark**: runs Read/Edit tasks (fix a bug, rename across
two files, read a value) through the real `OpenAICompatibleProvider`, in an
in-memory workspace with Claude Code's tool shapes and rules (absolute paths,
read before edit, unique `old_string`, unknown parameters rejected). Each
trial ends as:

| Outcome | What the router would see |
|---|---|
| `success` | Task verified |
| `wrong_result` / `max_steps` | Protocol fine, work wrong: silent quality loss |
| `fallback` | Refused before the first byte: the turn goes to Anthropic |
| `stream_error` | Stream broke after it started (e.g. invalid tool JSON): the client gets an `error` event and Claude Code retries; the retry goes to Anthropic, except on a conversation's first turn, where it is re-classified (ADR rule 4) |
| `truncated` | `max_tokens` or refusal |

```bash
npx tsx scripts/bench.ts --trials 5 --pad-kb 32
npx tsx scripts/bench.ts --task rename --model llama3.1:latest --json
```

`--pad-kb` grows the system prompt toward Claude Code's real context size.
Exit 0 when the success rate reaches `--min-success` (default 0.8).

**Anthropic passthrough smoke**: two small calls through the real
`AnthropicProvider` in `inject` mode (needs `ANTHROPIC_API_KEY`). Checks a
streamed turn (HTTP 200, `text/event-stream`, `anthropic-ratelimit-*` and
`request-id` forwarded, `message_start` ... `message_stop` in order, with
`accept-encoding` as Claude Code sends it) and that an upstream 4xx reaches the
client as Anthropic's own JSON error. Exit 0 = all checks pass.

```bash
npx tsx scripts/smoke-anthropic.ts --model claude-haiku-4-5
```

Groq and OpenRouter are both OpenAI-compatible: point `CHEAP_BASE_URL` /
`CHEAP_API_KEY` / `CHEAP_MODEL` at either one and run `bench.ts`, which already
streams through the real SSE translation.

## Context Injection Hook (`jev-router context`)

The `jev-router context` hook is a prompt injection hook designed for Claude Code. On every human prompt, it reads `prompt`, `session_id`, `cwd`, and `transcript_path` from `stdin` in JSON format, searches parallel sources (`ai-memory` wiki and `graphify` dependency graphs), ranks items using TypeSafe JEV System One (Stage 1 Choice and Stage 2 Noul-based relevance tests), and outputs prompt context injections.

### Hook Input Format (stdin)
The hook accepts a JSON object on standard input containing:
```json
{
  "prompt": "help me with standard routing",
  "session_id": "session-uuid",
  "cwd": "C:\\Dev\\JEV-APP-PC"
}
```

### Hook Output Format (stdout)
If matching context items are found and `CONTEXT_MODE=inject`, the hook outputs the following JSON to `stdout` containing the additional prompt context:
```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "[memória] título (path): trecho\n[grafo] projeto › comunidade: nós"
  }
}
```
If no items are found, or the mode is set to `shadow` or `off`, the output to `stdout` remains completely empty.

### Time budget and skips

Claude Code kills a `UserPromptSubmit` hook after its configured timeout (8 s here) and shows a warning, so the hook always fails open well before that:

- `CONTEXT_HOOK_BUDGET_MS` (default `5000`, clamped to 1000-7000) is the whole hook's budget, stdin included. When it runs out, the hook writes one line to stderr and exits 0 with nothing on stdout.
- The pipeline's own deadline is `min(CONTEXT_TIMEOUT_MS, budget - 1500 ms)`, so it normally gives up first and still records the run as `timeout`.
- When a JEV request fails (for example HTTP 529 or 503 while the API is overloaded), the requests still in flight are aborted at once, and the process exits even if the API never answers them. Before this fix such a request kept the hook alive for up to about 30 s.
- A `cwd` that is not a project (missing, a filesystem root, the home directory, or no `.git`, `.ai-memory.toml`, `graphify-out`, `package.json` or `CLAUDE.md` in it or an ancestor below home) is recorded as `skipped_cwd` without calling ai-memory, the graph scan or JEV.

### Bench history

Run `npx tsx scripts/bench.ts --trials 5 --pad-kb 32` before changing `CHEAP_MODEL`, and add a row.

| Date | Model | Trials | Success | Provider failures | Avg trial | Notes |
|---|---|---|---|---|---|---|
| 2026-10-04 | `gpt-oss:20b-cloud` | 15 (pad 32 KB, 12 steps) | 93% | 7% (1 `stream_error`) | 6.7 s | 25 schema errors, mostly `Read` with `offset: 0`, which Claude Code accepts (the bench now does too, so they were bench artifacts). A separate 45-trial capture: 0 `stream_error`, 1 `rename` `wrong_result` (import not updated). |
| 2026-10-07 | `gemma4:31b-cloud` (standard tier, gemma profile) | 6 (`rename` 3 + `fix-bug` 3, pad 16 KB, 12 steps) | 100% | 0% | 3.4 s | 0 tool errors. |
| 2026-10-07 | `gpt-oss:20b-cloud` (same run, for comparison) | 3 (`rename`, pad 16 KB) | 33% | 0% | 14.3 s | 1 `wrong_result` (import not updated), 1 `max_steps`, 3 schema errors (relative `file_path`, bad Grep `output_mode`). |
