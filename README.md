# jev-router

A local, Anthropic-compatible gateway for Claude Code (or any Messages-API
client). It uses **TypeSafe JEV** as a System One classifier to keep work off
the Claude quota. Each turn goes to the cheapest tier that can carry it:

- **Gemini Flash** for simple turns, through the Antigravity CLI (`agy`) on a
  Google subscription.
- **Gemini Argon** for medium reasoning, through the same `agy`.
- **Cursor** for coding turns, through the Cursor CLI on a Cursor
  subscription.
- **Claude**, byte for byte, for risky work and for everything the other
  tiers fail on.

```
                                          ┌─ simple ──────────▶ Flash   (agy)          ─┐
                                          │                                             │ on failure:
Claude Code ──▶ jev-router :8787 ──▶ JEV ─┼─ medium ──────────▶ Argon   (agy)          ─┤ next tier up,
                                          │    └─ coding ─────▶ Cursor  (cursor-agent) ─┤ Claude last
                                          │                                             │
                                          └─ risky / structural ▶ Claude (/v1/messages, byte for byte)

optional: local Ollama (OpenAI-compatible) before Flash or right before Claude (CHEAP_ROUTE)
```

The subscription tiers are CLI child processes, not APIs. Neither needs an API
key, and neither bills per token.

## How a turn is routed

### 1. Turns that never reach JEV

| Turn | Route | Reason in `x-jev-route` |
|---|---|---|
| `x-claude-code-request-class` in `ROUTER_PRIMARY_CLASSES` (default `auxiliary,compaction`) | Claude | `fallback:request-class` |
| Quota probe (`max_tokens <= 1`) | Claude | `classified` |
| Tool-result continuation (no new human text) | The tier that served the previous turn | `sticky` |
| A session that is already on Claude | Claude. JEV is only called for the Flash text-only check (see stickiness) | `sticky` |
| JEV fails or times out (`JEV_TIMEOUT_MS`, 1.5 s) | Claude | `fallback:classifier-failed` |

### 2. JEV's score picks the tier

A new human turn goes to JEV (`/v1/systemone`). One call returns:

- the distribution `{simple, standard, structural}`;
- three risk scores;
- `textOnly`, the probability that a written answer suffices without tools.

`src/domain/policy.ts` turns that into a tier deterministically:

| Condition, checked in order | Tier |
|---|---|
| `security_sensitive` or `destructive_or_production` ≥ 0.5 (**hard risk veto**) | Claude |
| `P(simple) ≥ ROUTER_MIN_CHEAP_PROBABILITY` (0.9), and `requires_inspection` < 0.5 | Flash |
| Argon on (`ARGON_TIER=on`, the default) | Argon, or Cursor for a coding turn |
| Argon off | Claude |

- **Argon or Cursor.** An Argon-scored turn goes to Cursor when all of these hold:
  - it is a coding turn: `textOnly < CURSOR_CODING_MAX_TEXT_ONLY` (0.75) and `standard + structural ≥ CURSOR_CODING_MIN_MASS` (0.5);
  - `CURSOR_TIER=on` and `CURSOR_CODING=on`;
  - the turn carries no image, document or forced `tool_choice`.
- **Argon profile.** `low`, `medium` or `high`:
  - `high`: `structural ≥ 0.45` or `requires_inspection ≥ 0.5`;
  - `medium`: `structural ≥ 0.3` or `standard ≥ 0.5`;
  - `low`: everything else.
- **Inspection veto.** `requires_inspection` keeps a turn off Flash, but not off Argon. It flags a verdict on the project's state that needs the repository investigated first ("is everything ok here?").
- **Inputs Flash can't take.** A Flash-scored turn with an image, a document or a forced `tool_choice` goes to Claude for that one turn (`gemini:ineligible-input`). The session stays on Flash for the next turn.
- **Middle band.** With Argon on, `ROUTER_MIN_STANDARD_PROBABILITY` is not used: Argon owns the middle band. With Argon off, `P(simple) + P(standard) ≥ 0.75` also goes to Flash.

### 3. Stickiness: a session only moves up

- **Tier rank:** Flash (0), Argon and Cursor (1), Claude (2).
- **A new human turn is re-classified and may escalate** (`escalated`). It never drops below the session's current tier.
- **Once on Claude, a session stays there:** going back down throws away the Anthropic prompt cache.
- **Exceptions to "stays on Claude":**
  - With Gemini on and `GEMINI_TIER_FROM_PRIMARY=true` (the default), a human turn of a Claude session is still scored. If JEV puts it on Flash with `textOnly ≥ GEMINI_TIER_MIN_TEXT_ONLY` (0.8), Flash answers it (`gemini:text-only`). This costs one extra JEV call per human turn of a Claude session.
  - At critical quota (below), a Claude session's next human turn is re-classified (`quota:reclassified`).
- **`/compact` does not reopen the cheaper tiers:** a Claude session stays on Claude (`pinned:compaction`).
- **Tool loops stay with the tier that started them.** A tool-result turn that follows a turn Claude served stays on Claude, even when the session's tier is Flash. Handing Claude's `tool_use` to another model would break the loop and the cache.

### 4. Failure only moves up

Each tier is one child-process hop behind its own circuit breaker. A breaker opens after `*_BREAKER_FAILURES` (3) consecutive failures and stays open for `*_BREAKER_COOLDOWN_MS` (5 min). An open or busy breaker skips the hop at once.

| Served by | On failure |
|---|---|
| Flash | → Argon (JEV's profile, then each higher profile) → [local Ollama, if `CHEAP_ROUTE=fallback`] → Claude |
| Argon | → the next higher profile → Claude |
| Cursor | → Argon (JEV's profile, then each higher profile) → Claude |
| Claude | → the 429/529 cascade below |

- **Argon is skipped when Flash failed on a timeout or a network error.** Both run through the same `agy`, and Argon never rescued such a turn; it only added its own 180 s limit.
- **When the turn reaches Claude after a failure,** the Claude profile is bumped one step (see "Claude profiles").
- **Silence limits:**
  - Flash: `TIER_FIRST_BYTE_TIMEOUT_MS` and `TIER_IDLE_TIMEOUT_MS`, 60 s each.
  - Argon: `ARGON_FIRST_BYTE_TIMEOUT_MS` and `ARGON_IDLE_TIMEOUT_MS`, 180 s each, because it reasons silently.
  - Cursor: `CURSOR_FIRST_BYTE_TIMEOUT_MS`, 90 s.
- **Keepalive.** While a hop is silent, the client receives an SSE `ping` every `KEEPALIVE_INTERVAL_MS` (15 s).

### 5. Quota substitution (`AGY_CLAUDE_SUBSTITUTE`)

When a subscription tier answers with its own quota error, the turn is not sent straight to Claude. It walks a substitution ladder first:

1. **Argon**, from the tier's profile up to `high`.
2. **A Claude model served by `agy`** (Claude through the Google subscription):
   - from Flash: `claude-sonnet-5-5-low`;
   - from Argon or Cursor: `claude-sonnet-5-5-medium`, `-high`, or `claude-opus-5-5-high`, matching the profile.
3. **Anthropic**, with a Claude profile no weaker than the substitute it replaced.

- `AGY_CLAUDE_MODEL_MAP` renames those model ids when `agy` lists them under other names.
- Served turns carry `x-jev-substituted-from`.
- The reason reads `quota-substitute:<from>-><to>`.
- `/healthz` → `models` shows each model's quota state.

### 6. Claude 429/529 cascade (`FAILOVER_ON_PRIMARY_RATE_LIMIT`, on by default)

When Claude itself answers 429 or 529:

1. **Short `retry-after`.** A `retry-after` of up to 10 s is waited out, and Claude is retried **once**: a retry keeps the prompt cache. A client that leaves during the wait starts nothing.
2. **Risk veto.** A turn under the hard risk veto gets the original error.
3. **Cursor, then Flash.** A turn with an image, a document or a forced `tool_choice` skips both.
4. **All tiers down.** The original Anthropic error reaches the client verbatim.

- **Quota probes are never rerouted.** Their 429 reaches Claude Code with its `anthropic-ratelimit-*` headers, so Claude Code can show its own quota warning.
- **The tier that took over keeps the session's tool loop.** A new human turn stays on that tier while the binding Claude quota window has not reset. After the reset, Claude is tried again.

### 7. Quota-aware routing (`QUOTA_ROUTING`, on by default)

Every Anthropic response carries the quota (`anthropic-ratelimit-unified-5h-*`, `-7d-*`). The router keeps the window closest to its limit.

- **From `QUOTA_PRESSURE` (80%):** the Flash text-only bar drops to `GEMINI_TIER_PRESSURE_MIN_TEXT_ONLY` (0.6). With Argon off, the Flash middle-band bar also drops, to `QUOTA_PRESSURE_MIN_STANDARD` (0.6).
- **From `QUOTA_CRITICAL` (95%):** a new human turn of a Claude session is re-classified and may leave Claude (`quota:reclassified`).
- **The hard risk veto still applies at every level.**

### 8. Claude profiles (`PRIMARY_PROFILES`, off by default)

When on, JEV's score also picks the Claude model and effort for a session's main loop, from `claude-sonnet-5-5` low up to `claude-opus-5-5` xhigh (`src/domain/policy.ts`, `primaryProfile`).

- A session that escalates to Claude after a subscription tier failed moves one step up the ladder.
- The profile only replaces Claude Code's default (Opus 5.5 at medium effort, not fast mode). An explicit `/model`, `/effort` or `/fast` always wins.
- If Anthropic refuses the profile, the client's own model is used until the next fresh session.
- When off, Claude gets whatever model the client picked.

### Consequences

- **Prompts leave the machine.** The subscription tiers send the conversation, including the code the agent reads, to Google (`agy`) or Cursor. Check their data policies before pointing a client repository at the router. The local Ollama hop keeps everything on your own server.
- **This setup is unsupported by Anthropic.** Claude Code is built for Claude models. On the other tiers, adaptive thinking, prompt caching and server tools (web search, tool search) are gone.
- **The subscription tiers are CLIs with their own agent loop, kept on a leash.**
  - The prompt tells the model to emit `<jev_tool_use>` blocks for Claude Code's tools instead of using its own.
  - `agy` runs from a router-owned profile (`GEMINI_TIER_HOME`) that denies every tool.
  - Cursor runs in `--mode ask`. Any side-effect tool call kills the process tree and counts as a miss (`tool attempt: …`); a short allowlist (todos, MCP tool listing) is tolerated.
  - The child processes are started without `GEMINI_API_KEY`/`GOOGLE_*` credentials, so they always use the subscription login.
- **Latency.** Measured on a Core i3 laptop with `jev-router stats`, 18 h on 2026-11-18:
  - Flash and Argon through `agy`: ~41 s on average.
  - Cursor: ~60 s, most of it the CLI booting Node.
  - Claude: ~7 s.
  - Answers from the CLI tiers arrive all at once, with simulated streaming.
- **Claude Code shows the wrong model name** on the other tiers. The truth is in the `x-jev-route` response header and in the `route decision` log line.

## Tier setup

### Flash and Argon (`agy`, Antigravity CLI)

1. Install `agy` and log in once interactively.
2. Set `GEMINI_TIER=on`.
3. Rebuild, or run `jev-router reload`.
4. Check that `/healthz` shows `gemini`, and `argon` too.

| Variable | Default | |
|---|---|---|
| `GEMINI_TIER` | `off` | Flash and the whole `agy` path |
| `GEMINI_TIER_MODEL` | `gemini-3.8-flash-high` | Flash model id |
| `GEMINI_TIER_BIN` | `%LOCALAPPDATA%\agy\bin\agy.exe` on Windows, `agy` elsewhere | |
| `GEMINI_TIER_TIMEOUT_MS` | `300000` | Hard limit per hop |
| `GEMINI_TIER_MIN_TEXT_ONLY` / `_PRESSURE_MIN_TEXT_ONLY` | `0.8` / `0.6` | Text-only bar for Claude sessions |
| `GEMINI_TIER_FROM_PRIMARY` | `true` | Score human turns of Claude sessions for Flash |
| `GEMINI_TIER_MAX_CONCURRENCY` | `1` | Busy = skip, never queue |
| `GEMINI_TIER_MAX_PROMPT_CHARS` | `3200000` | Older messages are omitted past it |
| `GEMINI_TIER_HOME` | `~/.jev-router/agy-home` | Isolated `agy` profile |
| `ARGON_TIER` | `on` | Needs `GEMINI_TIER=on` |
| `ARGON_MODEL_LOW` / `_MEDIUM` / `_HIGH` | `gemini-4-argon-low` / `-medium` / `-high` | Any `agy models` id, e.g. the Hydrogen profiles |
| `AGY_CLAUDE_SUBSTITUTE` | `on` | Claude-through-`agy` step of the quota ladder |

On Linux, `agy` can be a wrapper script that runs the binary on another host. This repository's author runs it over `ssh` because the laptop's CPU lacks an instruction `agy` needs. The router only needs a `GEMINI_TIER_BIN` that behaves like `agy`.

### Cursor (`cursor-agent`)

1. Install the Cursor CLI and log in once (`cursor-agent login`).
2. Set `CURSOR_TIER=on`.
3. Rebuild or reload, and check that `/healthz` shows `cursor`.

The runner starts the bundled Node directly:

- **Install layouts:**
  - Linux: `versions/<date>/node` + `index.js`, under `~/.local/share/cursor-agent`.
  - Windows: `node.exe`, under `%LOCALAPPDATA%\cursor-agent`.
- **Arguments:** `-p --mode ask --trust --output-format stream-json`.
- **The task goes in a `prompt.md` file** that Cursor reads. Text Cursor narrates before reading it is dropped from the answer.

| Variable | Default | |
|---|---|---|
| `CURSOR_TIER` | `off` | Routing tier and 429 cascade step |
| `CURSOR_TIER_BIN` | per platform (above) | Install root |
| `CURSOR_TIER_MODEL` | `auto` | |
| `CURSOR_TIER_TIMEOUT_MS` | `180000` | Hard limit per hop |
| `CURSOR_FIRST_BYTE_TIMEOUT_MS` | `90000` | |
| `CURSOR_HANG_COOLDOWN_MS` | `3600000` | Pause after a hung process |
| `CURSOR_TIER_MAX_PROMPT_CHARS` | `200000` | Over-budget turns skip Cursor without spawning it |
| `CURSOR_CODING` | `on` | Send coding turns to Cursor (otherwise Cursor is cascade-only) |
| `CURSOR_CODING_MAX_TEXT_ONLY` / `_MIN_MASS` | `0.75` / `0.5` | Coding-turn test |

### Local Ollama (`CHEAP_ROUTE`)

This is an optional hop to any OpenAI-compatible `/chat/completions` endpoint. It is meant for a model on your own GPU server, and it is off by default (`CHEAP_ROUTE=off`). It only ever sees turns the router already sent to Flash, so turns with images, documents, a forced `tool_choice` or a risk veto never reach it.

| Mode | Position |
|---|---|
| `first` | Before Flash. On failure: Flash → Argon → Claude |
| `fallback` | Once, after Flash and Argon both failed, right before Claude |

```bash
CHEAP_ROUTE=fallback
CHEAP_BASE_URL=http://<gpu-server>:11434/v1
CHEAP_API_KEY=<bearer the server expects>
CHEAP_MODEL=qwen3-coder:30b
CHEAP_TIMEOUT_MS=60000
```

- **Breaker.** The hop sits behind a breaker with the Gemini defaults, so a dead server costs at most three `CHEAP_TIMEOUT_MS` waits. An input the model cannot take, or a client abort, does not count as a failure.
- **Before turning it on:**
  - Check that the model calls tools (`ollama show <model>` lists `tools`): Claude Code always offers them.
  - Set `OLLAMA_CONTEXT_LENGTH` on the server, or Ollama silently truncates Claude Code's system prompt and tool schemas.
  - Measure the cold prefill. A ~24k-token prompt on a Tesla P40 takes about 68 s, which is why this repository runs `fallback` instead of `first`.
- **Telemetry.** Served turns show `route: cheap` (`cheap:local-first` / `cheap:local-fallback`). Failures appear as `upstream_failure = 'cheap:<code>'`.

**Translation for this hop: shallow, but never lossy.** Anthropic Messages are mapped 1:1 to Chat Completions:

| Anthropic | Chat Completions |
|---|---|
| `system` | `system` message |
| `tool_use` / `tool_result` | `tool_calls` / `role: tool` |
| `tools[].input_schema` | `functions[].parameters` |
| `tool_choice`, `stop_sequences` | `tool_choice`, `stop` |

- **Dropped on purpose:** `thinking` blocks, `cache_control`, server tools and deferred tools.
- **Anything else** (images, documents, `tool_reference`) throws `NotTranslatableError`, and the turn moves on to the next tier.
- **On the way back:**
  - text streams live;
  - tool calls are buffered, validated, and emitted as complete `tool_use` blocks;
  - tool call ids are sanitized to `^[a-zA-Z0-9_-]+$`.

## Repository structure

```
src/
├── cli.ts                         # `jev-router` binary: serve | reload | stats | statusline | context | agent-hook | help
├── serve.ts                       # composition root (only file that knows concrete classes)
├── supervisor.ts                  # reverse proxy over a forked worker: reload without dropping sessions
├── shutdown-report.ts             # stop-signal forensics: log at once, inspect recent processes, redact secrets
├── config.ts                      # zod-validated env → typed Config; fails fast at boot
├── quota.ts                       # anthropic-ratelimit-* headers → binding quota window and level
├── statusline.ts, agents.ts, hook-input.ts, build-info.ts, paths.ts, env.ts
├── domain/                        # pure, no I/O
│   ├── complexity.ts              # ComplexityDistribution (JEV levels)
│   └── policy.ts                  # selectTier, Argon profile, coding test, risk veto, Claude profiles
├── classifier/                    # System One port + adapters (JEV, offline heuristic)
├── routing/                       # decides; never talks to a provider
│   ├── router.ts                  # stickiness, escalation, cascade loops, profiles
│   ├── messages-body.ts           # open-schema parsing of the Anthropic body
│   ├── session-store.ts           # TTL + LRU map
│   ├── primary-profile.ts         # applies a Claude model + effort to the forwarded body
│   ├── bandit.ts                  # LinUCB for the cheap route; replay only, not wired into routing
│   └── read-first-hint.ts         # legacy, not wired into routing
├── providers/                     # executes; every provider answers in Anthropic format
│   ├── anthropic.ts               # byte-level forwarder (Claude)
│   ├── breaker.ts                 # per-tier circuit breaker + concurrency cap
│   ├── quota-state.ts             # per-model quota state for the substitution ladder
│   ├── cli-kill.ts                # kills a CLI child's whole process tree
│   ├── gemini/                    # agy runner, isolated profile, prompt rendering, tool-call parsing
│   ├── cursor/                    # cursor-agent runner + provider
│   └── openai/                    # OpenAI-compatible provider + translation (local Ollama hop)
├── proxy/
│   ├── server.ts                  # Fastify routes; tier hops, fallbacks and the 429 cascade
│   └── headers.ts                 # hop-by-hop filtering, credential extraction
├── context/                       # `jev-router context` hook: ai-memory + graphify, ranked by JEV
└── telemetry/                     # cost audit; can never fail or slow a response
```

The router returns a decision (tier + reason) and knows nothing about base
URLs, headers, wire formats or child processes. Providers own all of that.
Adding a provider never touches routing rules, and routing rules are tested
without HTTP.

## Dependencies

| Package | Why |
|---|---|
| `fastify` | Raw-buffer bodies with a hard `bodyLimit`, streamed replies, pino logging |
| `undici` | One HTTP client for the upstreams. `request()` exposes the raw, non-decompressed stream that the Anthropic passthrough needs |
| `zod` | Validates env, the body fields we read, and every JEV/OpenAI payload |

Deliberately absent:

- `dotenv`: `process.loadEnvFile` does the job.
- The Anthropic and OpenAI SDKs: a gateway forwards bytes; it doesn't build requests.
- An LRU library and an ORM.
- A SQLite driver package: the cost audit uses Node's built-in `node:sqlite`. Node 22 and 24 still flag it experimental and print one `ExperimentalWarning` at startup.

## Setup

```bash
git clone https://github.com/Covre81/jev-router.git && cd jev-router
npm ci
cp .env.example .env
#   TYPESAFE_API_KEY=...       (or CLASSIFIER=heuristic for offline dev)
#   GEMINI_TIER=on             after logging in to agy once
#   CURSOR_TIER=on             after `cursor-agent login`
npm test                       # fake upstreams, no network
npm run build && npm start
curl -s localhost:8787/healthz
```

Requires Node ≥ 22.19.

### Global CLI (`npm link`)

```bash
npm link                   # puts `jev-router` on your PATH
mkdir -p ~/.jev-router && cp .env.example ~/.jev-router/.env   # then fill in the keys
jev-router serve           # start the gateway (default command)
jev-router stats --since 24h
```

The env lookup order is:

1. variables already set in the shell;
2. `--env <file>`;
3. `./.env`;
4. `~/.jev-router/.env`.

`JEV_ROUTER_HOME` moves the `~/.jev-router` directory.

On Linux, the router can run as a `systemctl --user` service whose `ExecStart` is `node dist/cli.js serve --env <checkout>/.env`.

### Reload without dropping sessions

Every Claude Code session goes through the router, so a plain restart would cut them all. `jev-router serve` therefore runs as a **supervisor**:

- It owns `HOST:PORT`.
- It forks the real gateway as a worker on an ephemeral loopback port.
- It pipes every request to the worker unbuffered, so SSE streams flow as written.

```bash
npm run build              # runs `reload` itself via the `postbuild` hook
jev-router reload          # manual, e.g. after editing .env
```

`reload` asks the supervisor to start a fresh worker from the new `dist/`. The request goes through a loopback-only control port, `CONTROL_PORT` (default `PORT + 1`).

- **The swap.** Once the new worker listens, new requests go to it. The old worker finishes its open streams and is cut after `RELOAD_DRAIN_MS` (2 min).
- **What reload picks up.** The worker re-reads the env file, so `.env` changes apply on `reload`. Two kinds of change need a full restart instead:
  - `HOST`, `PORT`, `CONTROL_PORT` and `RELOAD_DRAIN_MS`, which belong to the supervisor;
  - edits to `supervisor.ts` itself.
- **A crashed worker** is restarted after 1, 2, 4, 4 and 4 s. After that the supervisor answers 503 and waits for a `reload`.

`/healthz` reports which build answers, and the state of each tier:

```
{ok, sha, builtAt, startedAt, pid, quota,
 gemini: {model, breaker, inFlight}, argon: {breaker, inFlight},
 cursor: {model, breaker, inFlight}, cheap: {model, position, breaker, inFlight},
 models: {<model id>: {state}}}
```

- A tier that is off shows `null`.
- `quota` is the last Claude quota reading, or `null` before the first Anthropic response.
- For `npm run dev` (tsx watch), set `SUPERVISOR_ENABLED=false`.

**Claude credentials** (`UPSTREAM_AUTH_MODE`):

- `passthrough` (default): the proxy holds no Anthropic secret and forwards whatever Claude Code sends, API key or claude.ai login.
- `inject`: the proxy holds `ANTHROPIC_API_KEY` and replaces the client's credentials. Off loopback, `PROXY_AUTH_TOKEN` is required.

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

- **`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` is required.** Without it, Claude Code doesn't send `x-claude-code-request-class`. Its side requests (titles, classifiers, summaries) then get classified like any prompt, and sessions lose their stable key.
- **With a claude.ai login,** the Claude route keeps using your subscription.
- **Other clients:** point the base URL at `http://127.0.0.1:8787`. Without session headers, conversations are keyed by a hash of their first message.

**Status line.** To see under every Claude Code prompt whether the router is up and where the session's last turn went:

```json
{
  "statusLine": {
    "type": "command",
    "command": "node /path/to/jev-router/dist/cli.js statusline --env /path/to/jev-router/.env"
  }
}
```

It prints a line like `jev-router ✓ · last: <route> <model> (JEV 0.93) · today 12/40 cheap`.

| Marker | Meaning |
|---|---|
| `jev-router ✗ offline` | `/healthz` didn't answer within 500 ms |
| `⚠ build velho` | `dist/build-info.json` is newer than the build that answers. Run `jev-router reload` |
| `last: …` | The session's last route, its model, and JEV's P(simple) or the route reason |
| `today N/M cheap` | Today's routed turns (M), and how many of them the local Ollama hop served (N). Flash, Argon and Cursor turns are not counted in N |
| `ctx 125k` | The context the main agent sent on its last request. From 200k on it turns into `⚠ ctx … → /compact or /clear` |
| `cota 82% 5h` | The binding Claude quota window. Only with `STATUSLINE_QUOTA=1` |
| `🤖 2 ⏳ 1h05` | Running subagents and the age of the oldest (needs the agent hook below) |

- Pass `--env` explicitly: the status line runs in each project's directory, and that project's own `.env` could set another `PORT`.
- `--json` prints `{"text": …, "level": "ok" | "warn" | "critical"}` for a wrapping status line.

**Agent hook.** The subagent counter needs `jev-router agent-hook` on four hook events:

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

How it tracks subagents:

- **SubagentStart** writes an empty file `~/.jev-router/agents/<session>/<agent>`; its mtime is the start time.
- **SubagentStop** deletes that file.
- **SessionEnd** deletes the session's folder.
- **SessionStart** cleans up after crashes, since a crashed Claude Code fires neither stop event. On `resume` it drops the session's agents. On any start it deletes session folders untouched for 7 days.

The hook always exits 0, so it can never block a subagent from stopping.

## Telemetry and cost audit

Every `/v1/messages` exchange is written to `~/.jev-router/telemetry.db`, in the table `router_logs`. Override the path with `TELEMETRY_DB_PATH`; turn it off with `TELEMETRY_ENABLED=false`.

| Column | Meaning |
|---|---|
| `session_id` | Conversation key: `<x-claude-code-session-id>:<agent>`, or a hash of the first message |
| `human_prompt_hash` | sha256 of the human-typed text; null on tool-result turns. Equal hashes = repeated prompt |
| `jev_decision` | JSON of JEV's answer: distribution, risk scores, `textOnly`, the classifier's latency and token usage. Null when JEV was not asked |
| `final_provider` | `anthropic`, `gemini` (Flash, Argon and Claude-through-`agy`), `cursor`, or `openai` (local Ollama) |
| `model`, `route_reason` | The model that answered, and the reason shown in `x-jev-route` |
| `tokens_in` / `tokens_out` / `cache_read_tokens` | From the response `usage`; `tokens_in` includes cache reads and writes |
| `latency_ms` | Request received → last byte sent to the client |
| `outcome` | `ok`, `http_error`, `stream_error`, `client_abort` or `proxy_error` |
| `upstream_status`, `upstream_ratelimit_headers`, `upstream_failure` | The upstream's status, the Anthropic rate-limit header names, and the failure reason |
| `prev_tool_errors`, `prev_interrupted` | The user's reaction to the **previous** turn of the session: `is_error` tool results and Claude Code's `[Request interrupted by user…]` |

- `upstream_failure` holds the last failed hop. Examples: `cheap:<code>` for the local Ollama, and `retry-after:ok` for a 429 that the single retry cleared.
- `stats` joins the reaction columns to the provider that served the previous turn.

**Why it adds no latency:**

- The response body passes through a tap that copies nothing. It only scans `data:` lines that mention `usage` or `error`, and decodes compressed bodies on a side channel.
- The row is built after the client socket closes and goes into an in-memory queue.
- A timer writes the queue to SQLite once a second, in one transaction. If a write fails, the rows stay queued and the response is unaffected.

**`jev-router stats`** prints, per provider:

- requests and successful answers;
- tokens, cache reads and average latency;
- fallback counters (Flash → Claude, Cursor → next tier, local Ollama → next tier);
- the follow-up reaction table;
- the estimated Anthropic tokens saved and the **net cost in dollars** against a baseline of the same traffic on Anthropic with a warm prompt cache.

On a Claude subscription, those dollars are API-equivalent quota, not a bill.

- **Flags:** `--since 24h|7d`, `--json` and `--by-profile` (main-loop calls per session, with cache hit and USD by model and effort).
- **More flags with `TELEMETRY_DIAGNOSTICS=true`:** `--by-class`, `--cache-misses` and `--daily`. Diagnostics store only hashes, lengths, token counts and header names, never prompt text.
- **Prices:**
  - Anthropic: list prices per model from a built-in table, overridable with `PRIMARY_PRICE_*`.
  - Subscription tiers: $0.
  - The local Ollama hop: `CHEAP_PRICE_*_PER_MTOK`, which stays unset for a self-hosted model.
  - JEV: `JEV_PRICE_*_PER_MTOK`, from your TypeSafe plan.

## Checks against real APIs (local only)

CI runs only the offline suite (`npm test`), where every upstream is a local fake. The scripts below call real services and are run by hand. They read the same env files as the CLI.

| Script | What it checks |
|---|---|
| `npx tsx scripts/test-jev-real.ts` | One real `/v1/systemone` call, checked strictly against the adapter's schema |
| `npx tsx scripts/eval-jev.ts` | JEV's routing scores against a labeled prompt set |
| `npx tsx scripts/bench.ts --trials 5 --pad-kb 32` | Read/Edit tool tasks through the real OpenAI-compatible provider. Run it before pointing `CHEAP_MODEL` at a new model |
| `npx tsx scripts/smoke-anthropic.ts --model claude-haiku-4-5` | Two small calls through the real `AnthropicProvider` in `inject` mode (needs `ANTHROPIC_API_KEY`) |
| `npx tsx scripts/bandit-replay.ts` | Replays the cheap-route history through the LinUCB bandit, read-only |

`bench.ts` outcomes:

| Outcome | What the router would see |
|---|---|
| `success` | Task verified |
| `wrong_result` / `max_steps` | Protocol fine, work wrong: silent quality loss |
| `fallback` | Refused before the first byte: the turn moves on |
| `stream_error` | Stream broke after it started |
| `truncated` | `max_tokens` or refusal |

## Context injection hook (`jev-router context`)

A `UserPromptSubmit` hook for Claude Code. On every human prompt it:

1. reads `prompt`, `session_id`, `cwd` and `transcript_path` from stdin;
2. searches the `ai-memory` wiki and the `graphify` graphs under `CONTEXT_GRAPH_ROOT`;
3. ranks the candidates with JEV (a Stage 1 choice, then a Stage 2 relevance test);
4. prints the winners as `additionalContext`.

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "[memória] título (path): trecho\n[grafo] projeto › comunidade: nós"
  }
}
```

The output is empty when nothing matches, or when `CONTEXT_MODE` is `shadow` or `off`.

**Time budget.** Claude Code kills a hook after its timeout (8 s here), so this hook fails open well before that:

- `CONTEXT_HOOK_BUDGET_MS` (default `5000`, clamped to 1000–7000) is the whole budget, stdin included. When it runs out, the hook writes one line to stderr and exits 0 with nothing on stdout.
- The pipeline's own deadline is `min(CONTEXT_TIMEOUT_MS, budget - 1500 ms)`, so it normally gives up first and records the run as `timeout`.
- A failed JEV request aborts the requests still in flight at once.
- **Skipped `cwd`s.** A `cwd` that is not a project is recorded as `skipped_cwd`, without calling ai-memory, the graph scan or JEV. That covers:
  - a missing directory, a filesystem root, or the home directory;
  - a directory where neither it nor an ancestor below home has `.git`, `.ai-memory.toml`, `graphify-out`, `package.json` or `CLAUDE.md`.
