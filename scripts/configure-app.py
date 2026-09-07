#!/usr/bin/env python3
"""Wires a generated create-context-graph app up for this workshop.

Run this once, right after `uvx create-context-graph` finishes, before
`make install`. Same for every domain: makes `make start` work from inside
a Codespace, not just on localhost. sentence-transformers (embeddings) stays
on -- the Codespace prebuild warms its wheel and model cache, so `make
install` still finishes quickly without capping functionality to reach that
speed. GLiNER and spaCy are both disabled (see below) -- unused by this
workshop, and both cost real install/runtime time for nothing.

Usage: python3 configure-app.py [path-to-generated-app]  (defaults to ".")
"""
import re
import sys
from pathlib import Path

app_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

TOTAL_CHECKS = 8


def _ok(n: int) -> None:
    print(f"{n}/{TOTAL_CHECKS} checks. {n}/{TOTAL_CHECKS} complete [ok].")


def _fail(n: int, message: str) -> None:
    sys.exit(f"{n}/{TOTAL_CHECKS} failed: {message}")


# --- Copy OpenAI credentials from this repo's own .env ----------------------
#
# Both this repo's own `.vscode/mcp.json` (Copilot's memory server, task 6)
# and the generated app need OPENAI_API_KEY and MEMORY_EMBEDDING. Learners
# used to paste them a second time, straight into the generated app's own
# .env, right after this script ran. This repo's own .env is now the one
# place they're entered (see .env.example) -- copy them into the generated
# app's .env instead of asking twice.

WORKSHOP_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
app_env_path = app_dir / ".env"


def _read_env(path: Path) -> dict:
    values: dict = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _set_env_line(src: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    line = f"{key}={value}"
    if pattern.search(src):
        return pattern.sub(line, src, count=1)
    return src.rstrip("\n") + f"\n{line}\n"


if not WORKSHOP_ENV_PATH.exists():
    _fail(
        1,
        f"{WORKSHOP_ENV_PATH} not found — fill in this repo's own "
        ".env (copy .env.example) with your Aura credentials, "
        "OPENAI_API_KEY, and MEMORY_EMBEDDING before running this script.",
    )

workshop_env = _read_env(WORKSHOP_ENV_PATH)
openai_api_key = workshop_env.get("OPENAI_API_KEY", "")
memory_embedding = workshop_env.get("MEMORY_EMBEDDING", "")
openai_default_model = workshop_env.get("OPENAI_DEFAULT_MODEL", "")

if not openai_api_key:
    _fail(
        1,
        f"OPENAI_API_KEY is not set in {WORKSHOP_ENV_PATH} — add it "
        "there (not in the generated app's own .env) and rerun.",
    )

if not app_env_path.exists():
    _fail(1, f"{app_env_path} not found — did create-context-graph run?")

app_env_src = app_env_path.read_text()

if (
    f"OPENAI_API_KEY={openai_api_key}" in app_env_src
    and (not memory_embedding or f"MEMORY_EMBEDDING={memory_embedding}" in app_env_src)
    and (not openai_default_model or f"OPENAI_DEFAULT_MODEL={openai_default_model}" in app_env_src)
):
    _ok(1)
else:
    app_env_src = _set_env_line(app_env_src, "OPENAI_API_KEY", openai_api_key)
    if memory_embedding:
        app_env_src = _set_env_line(app_env_src, "MEMORY_EMBEDDING", memory_embedding)
    if openai_default_model:
        app_env_src = _set_env_line(app_env_src, "OPENAI_DEFAULT_MODEL", openai_default_model)
    app_env_path.write_text(app_env_src)
    _ok(1)


# --- Wire up Codespaces networking ------------------------------------------

makefile_path = app_dir / "Makefile"
makefile_src = makefile_path.read_text()

VARS_ANCHOR = "-include .env\nexport\n"
VARS_BLOCK = """-include .env
export

# Codespaces forwards ports on their own subdomain, not localhost — compute
# the right URLs so `make start` works unmodified in a Codespace or locally.
ifdef CODESPACE_NAME
FRONTEND_API_URL := https://$(CODESPACE_NAME)-$(BACKEND_PORT).$(GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN)/api
FRONTEND_ORIGIN := https://$(CODESPACE_NAME)-$(FRONTEND_PORT).$(GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN)
else
FRONTEND_API_URL := http://localhost:$(BACKEND_PORT)/api
FRONTEND_ORIGIN := http://localhost:$(FRONTEND_PORT)
endif
"""

OLD_BACKEND = "dev-backend:\n\tcd backend && uv run uvicorn app.main:app --reload --port $${BACKEND_PORT:-8000}"
NEW_BACKEND = "dev-backend:\n\tcd backend && CORS_ORIGINS=$(FRONTEND_ORIGIN) uv run uvicorn app.main:app --reload --port $${BACKEND_PORT:-8000}"

OLD_FRONTEND = "dev-frontend:\n\tcd frontend && NEXT_PUBLIC_API_URL=http://localhost:$${BACKEND_PORT:-8000}/api npm run dev -- --port $${FRONTEND_PORT:-3000}"
NEW_FRONTEND = "dev-frontend:\n\tcd frontend && NEXT_PUBLIC_API_URL=$(FRONTEND_API_URL) npm run dev -- --port $${FRONTEND_PORT:-3000}"

if VARS_BLOCK in makefile_src:
    _ok(2)
elif VARS_ANCHOR not in makefile_src or OLD_BACKEND not in makefile_src or OLD_FRONTEND not in makefile_src:
    _fail(
        2,
        "Makefile doesn't match what this script expects — the "
        "generator may have changed. Ask whoever is running the workshop "
        "for help.",
    )
else:
    makefile_src = makefile_src.replace(VARS_ANCHOR, VARS_BLOCK, 1)
    makefile_src = makefile_src.replace(OLD_BACKEND, NEW_BACKEND, 1)
    makefile_src = makefile_src.replace(OLD_FRONTEND, NEW_FRONTEND, 1)
    makefile_path.write_text(makefile_src)
    _ok(2)


# --- Make ports public as soon as they exist, not before -------------------
#
# devcontainer.json used to run `gh codespace ports visibility ...` as
# postAttachCommand. That fires at attach — before `make start` has ever run
# — so neither port is forwarded yet and the call has nothing to act on.
# `gh codespace ports` confirmed this live: it omitted port 8000 entirely
# until something touched it via the Ports UI, even with uvicorn already
# listening. The one point both ports reliably exist is right after `make
# start` launches them, so the visibility call moves here and retries for a
# few seconds to cover the registration lag.

makefile_path = app_dir / "Makefile"  # re-read; may have just been rewritten
makefile_src = makefile_path.read_text()

OLD_START_BODY = (
    "\t@trap 'kill 0' EXIT; \\\n"
    "\t\t$(MAKE) dev-backend & \\\n"
    "\t\t$(MAKE) dev-frontend & \\\n"
    "\t\twait"
)
NEW_START_BODY = (
    "\t@trap 'kill 0' EXIT; \\\n"
    "\t\t$(MAKE) dev-backend & \\\n"
    "\t\t$(MAKE) dev-frontend & \\\n"
    "\t\t$(MAKE) publish-ports & \\\n"
    "\t\twait"
)

PUBLISH_PORTS_TARGET = """
# Codespaces only lets you set a port's visibility once it has actually
# forwarded a live listener — retry for up to 30s to cover the gap between
# `make start` launching the servers and Codespaces registering the ports.
publish-ports:
	@if [ -n "$$CODESPACE_NAME" ]; then \\
		for i in $$(seq 1 30); do \\
			gh codespace ports visibility $(BACKEND_PORT):public $(FRONTEND_PORT):public -c "$$CODESPACE_NAME" >/dev/null 2>&1 && break; \\
			sleep 1; \\
		done; \\
	fi
"""

if PUBLISH_PORTS_TARGET.strip() in makefile_src:
    _ok(3)
elif OLD_START_BODY not in makefile_src:
    _fail(
        3,
        "Makefile's start target doesn't match what this script "
        "expects — the generator may have changed. Ask whoever is running "
        "the workshop for help.",
    )
else:
    makefile_src = makefile_src.replace(OLD_START_BODY, NEW_START_BODY, 1)
    makefile_src = makefile_src.replace(
        "dev: start", "dev: start" + PUBLISH_PORTS_TARGET, 1
    )
    makefile_path.write_text(makefile_src)
    _ok(3)


# --- Drop the spaCy model download from install-backend --------------------
#
# Pinning this to `--direct` (skipping spaCy's live compatibility-check
# network call) fixed the cache-hit itself -- confirmed no re-download --
# but the wait persisted anyway: `uv run python -c "import spacy"` and
# `uv run python -m spacy download ...` are two separate `uv run`
# invocations, each paying spaCy's own cold-import cost (thinc/blis/cython
# extension loading) on top of `uv run`'s environment-sync check against
# the full torch/transformers tree. None of that is network-bound, so no
# cache fixes it. spaCy is unused at runtime (see the ExtractionConfig
# patch below) -- dropping the step entirely is the only way to actually
# remove the wait, not just make it reliably fast rather than reliably
# hung.

makefile_path = app_dir / "Makefile"
makefile_src = makefile_path.read_text()

OLD_INSTALL_BACKEND_VARIANTS = [
    (
        'install-backend:\n'
        '\tcd backend && uv sync --extra dev\n'
        '\t@cd backend && uv run python -c "import spacy" 2>/dev/null && \\\n'
        '\t\tuv run python -m spacy download en_core_web_sm || \\\n'
        '\t\techo "spacy not installed — skipping model download (NAMS backend or extraction extra disabled)"'
    ),
    (
        'install-backend:\n'
        '\tcd backend && uv sync --extra dev\n'
        '\t@cd backend && uv run python -c "import spacy" 2>/dev/null && \\\n'
        '\t\tuv run python -m spacy download en_core_web_sm-3.8.0 --direct || \\\n'
        '\t\techo "spacy not installed — skipping model download (NAMS backend or extraction extra disabled)"'
    ),
]
NEW_INSTALL_BACKEND = (
    'install-backend:\n'
    '\tcd backend && uv sync --extra dev'
)

matched = next((v for v in OLD_INSTALL_BACKEND_VARIANTS if v in makefile_src), None)
if NEW_INSTALL_BACKEND in makefile_src and matched is None:
    _ok(4)
elif matched is None:
    _fail(
        4,
        "Makefile's install-backend target doesn't match what this "
        "script expects — the generator may have changed. Ask whoever is "
        "running the workshop for help.",
    )
else:
    makefile_src = makefile_src.replace(matched, NEW_INSTALL_BACKEND, 1)
    makefile_path.write_text(makefile_src)
    _ok(4)


# --- Frozen npm install from the prebuild's lockfile ------------------------
#
# Plain `npm install` with no lockfile doesn't just fetch tarballs -- it
# re-resolves the whole dependency graph against the registry, runs a live
# audit call, and pings npm's update-notifier. Measured identical: `npm
# install` ~10s, `npm ci` off the same warmed cache ~6s, with no
# update-notifier ping. warm-cache.sh resolves this exact dependency set
# once and persists the resulting lockfile to
# /workspaces/.frontend-package-lock.json (captured in the same prebuild
# snapshot as the caches) -- copy it in here and switch to `npm ci`, a
# frozen install with no resolution step. Falls back to `npm install` when
# no warmed lockfile is present (e.g. running outside this Codespace setup).

WARM_LOCKFILE = Path("/workspaces/.frontend-package-lock.json")
frontend_lockfile = app_dir / "frontend" / "package-lock.json"

OLD_INSTALL_FRONTEND = "install-frontend:\n\tcd frontend && npm install"
NEW_INSTALL_FRONTEND = "install-frontend:\n\tcd frontend && npm ci --no-audit --no-fund"

if not WARM_LOCKFILE.exists():
    _ok(5)
elif frontend_lockfile.exists():
    _ok(5)
else:
    makefile_path = app_dir / "Makefile"
    makefile_src = makefile_path.read_text()
    if NEW_INSTALL_FRONTEND in makefile_src:
        _ok(5)
    elif OLD_INSTALL_FRONTEND not in makefile_src:
        _fail(
            5,
            "Makefile's install-frontend target doesn't match what "
            "this script expects — the generator may have changed. Ask "
            "whoever is running the workshop for help.",
        )
    else:
        frontend_lockfile.write_text(WARM_LOCKFILE.read_text())
        makefile_src = makefile_src.replace(OLD_INSTALL_FRONTEND, NEW_INSTALL_FRONTEND, 1)
        makefile_path.write_text(makefile_src)
        _ok(5)


# --- Fix broken conversation history (create-context-graph bug) ------------
#
# agent.py.j2 (OpenAI Agents SDK framework) does `context.get("messages", [])`,
# but MemoryIntegration.get_context() returns {"session_id", "context",
# "has_context"} -- there is no "messages" key. history is silently [] on
# every single turn, so the agent never sees prior conversation. The real
# formatted history string is right there under context["context"]; this
# patches the generated agent.py to use it. Upstream fix pending -- can't
# land in time for v0.14.0 (already released), so this repairs the scaffold
# post-generation instead of waiting for the next PyPI release.

agent_path = app_dir / "backend" / "app" / "agent.py"

OLD_HISTORY = '''    context = await get_context(session_id, query=message)
    history = context.get("messages", [])

    # Build input with structured conversation history
    if history:
        history_block = "\\n\\n".join(
            f"[{m['role'].upper()}]\\n{m['content']}" for m in history
        )
        input_message = (
            f"<conversation_history>\\n{history_block}\\n</conversation_history>\\n\\n"
            f"[USER]\\n{message}"
        )
    else:
        input_message = message'''

NEW_HISTORY = '''    context = await get_context(session_id, query=message)
    history_text = context.get("context", "") if isinstance(context, dict) else ""

    if history_text:
        input_message = (
            f"<conversation_history>\\n{history_text}\\n</conversation_history>\\n\\n"
            f"[USER]\\n{message}"
        )
    else:
        input_message = message'''

if not agent_path.exists():
    _ok(6)
else:
    agent_src = agent_path.read_text()
    occurrences = agent_src.count(OLD_HISTORY)
    if occurrences == 0 and NEW_HISTORY in agent_src:
        _ok(6)
    elif occurrences != 2:
        _fail(
            6,
            "backend/app/agent.py doesn't match what this script expects "
            f"(found {occurrences} of 2 expected occurrences) — the generator may "
            "have changed. Ask whoever is running the workshop for help.",
        )
    else:
        agent_path.write_text(agent_src.replace(OLD_HISTORY, NEW_HISTORY))
        _ok(6)


# --- Disable GLiNER and spaCy extraction ------------------------------------
#
# GLiNER's model load has correlated, twice, with the backend process exiting
# mid-request during this workshop (right after "Fetching N files ...
# Reconstruction complete" in the uvicorn log, no traceback). Root cause not
# proven (dmesg is blocked in this container; OOM ruled out separately), but
# it's a reproducible enough pattern to disable for the workshop rather than
# risk it live.
#
# spaCy is disabled too -- this workshop's demo data and prompts never
# exercise it, and `en_core_web_sm` is a live download (a GitHub release
# asset, not a registry package uv's cache can warm the same way) that added
# real time to every `make install` for a stage nothing here uses. LLM
# fallback extraction still runs.

memory_path = app_dir / "backend" / "app" / "memory.py"

OLD_IMPORT = "    from neo4j_agent_memory import MemorySettings, NamsConfig"
NEW_IMPORT = "    from neo4j_agent_memory import ExtractionConfig, MemorySettings, NamsConfig"

OLD_COMMON = '''    common: dict = {}
    if llm_model:
        common["llm"] = llm_model
    if embedding_model:
        common["embedding"] = embedding_model'''

NEW_COMMON = '''    common: dict = {}
    if llm_model:
        common["llm"] = llm_model
    if embedding_model:
        common["embedding"] = embedding_model
    common["extraction"] = ExtractionConfig(enable_gliner=False, enable_spacy=False)'''

if not memory_path.exists():
    _ok(7)
else:
    memory_src = memory_path.read_text()
    if NEW_COMMON in memory_src:
        _ok(7)
    elif OLD_IMPORT not in memory_src or OLD_COMMON not in memory_src:
        _fail(
            7,
            "backend/app/memory.py doesn't match what this script expects "
            "— the generator may have changed. Ask whoever is running the "
            "workshop for help.",
        )
    else:
        memory_src = memory_src.replace(OLD_IMPORT, NEW_IMPORT, 1)
        memory_src = memory_src.replace(OLD_COMMON, NEW_COMMON, 1)
        memory_path.write_text(memory_src)
        _ok(7)


# --- Teach the agent its own tools, including that it IS the graph --------
#
# The agent already has run_cypher (arbitrary read-only Cypher) and
# get_graph_schema() (lists node labels/relationship types) -- it can query
# :Conversation/:Message nodes right now, no new tool needed. But nothing in
# its prompt tells it those exist, or that a schema check should happen
# before answering, so it falls back to plausible-sounding priors ("chat
# history isn't graph data") instead of using tools it already has. Verified
# live: asked directly, the agent claimed conversation isn't graph data --
# false, confirmed by querying the same database directly.
#
# Splits the rendered SYSTEM_PROMPT into its two constant halves (this exact
# IMPORTANT/CRITICAL suffix is hardcoded in agent.py.j2, not templated per
# domain) to isolate the domain text in between without hardcoding it.

agent_path = app_dir / "backend" / "app" / "agent.py"

PROMPT_PREFIX = 'SYSTEM_PROMPT = """'
PROMPT_SUFFIX = '''

IMPORTANT: You MUST use the available tools to query the knowledge graph before answering any question about the data. Never guess or make up information — always use tools to look up actual data from the graph.

CRITICAL: Call tools DIRECTLY without any introductory text. Do NOT say "I'll search for..." or "Let me look up..." before calling a tool — just call the tool immediately. Only generate text AFTER you have received the tool results and are ready to provide your final answer."""'''

SYSTEM_WIDE_PREAMBLE = """Before answering any question, always call get_graph_schema first, then call at least one more tool to get real data — never answer from memory or guesswork.

Tools available in every domain:
- get_graph_schema(): Returns the graph's node labels and relationship types. Call this first, every time, so you know what actually exists before querying it.
  Example: at the start of any question, to see what's in the graph before searching it.
- run_cypher(query, parameters): Executes an arbitrary read-only Cypher query against the whole graph, not just domain data. Conversation messages, entities, and reasoning traces you and the user produce are stored as nodes in this same graph (:Conversation, :Message, :Entity, linked via HAS_MESSAGE, MENTIONS, etc.) and are queryable the same way as any other data.
  Example: "is our conversation stored in the graph?" -> run_cypher a query matching :Conversation/:Message nodes.
"""

DOMAIN_TOOLS_HEADER = "# Agent tools"
DOMAIN_TOOLS_FOOTER = "async def run_cypher"
TOOL_DEF_PATTERN = re.compile(
    r'async def (\w+)\(([^)]*)\) -> str:\n    """(.+?)"""', re.DOTALL
)


def build_domain_tool_map(agent_src: str) -> str:
    """Parse the actual generated domain tools (name/params/docstring) instead
    of hardcoding one domain's tool names -- this runs the same regardless of
    which --domain the wizard was given."""
    start = agent_src.index(DOMAIN_TOOLS_HEADER)
    end = agent_src.index(DOMAIN_TOOLS_FOOTER)
    domain_block = agent_src[start:end]
    lines = []
    for name, params, doc in TOOL_DEF_PATTERN.findall(domain_block):
        param_names = [p.split(":")[0].strip() for p in params.split(",") if p.strip()]
        sig = ", ".join(param_names)
        lines.append(f"- {name}({sig}): {doc.strip()}.")
    tool_lines = "\n".join(lines)
    return f"""
Tools specific to this domain:
{tool_lines}

Always call get_graph_schema first, then call at least one of the tools above to answer with real data from the graph."""


if not agent_path.exists():
    _ok(8)
else:
    agent_src = agent_path.read_text()
    if SYSTEM_WIDE_PREAMBLE in agent_src:
        _ok(8)
    elif PROMPT_PREFIX not in agent_src or PROMPT_SUFFIX not in agent_src:
        _fail(
            8,
            "backend/app/agent.py SYSTEM_PROMPT doesn't match what this "
            "script expects — the generator may have changed. Ask whoever is "
            "running the workshop for help.",
        )
    else:
        domain_tool_map = build_domain_tool_map(agent_src)
        prefix_idx = agent_src.index(PROMPT_PREFIX) + len(PROMPT_PREFIX)
        suffix_idx = agent_src.index(PROMPT_SUFFIX, prefix_idx)
        domain_text = agent_src[prefix_idx:suffix_idx]
        new_prompt_body = (
            SYSTEM_WIDE_PREAMBLE + "\n" + domain_text.rstrip() + "\n" + domain_tool_map
        )
        agent_src = (
            agent_src[:prefix_idx]
            + new_prompt_body
            + PROMPT_SUFFIX
            + agent_src[suffix_idx + len(PROMPT_SUFFIX):]
        )
        agent_path.write_text(agent_src)
        _ok(8)


print("Config complete. Continue with: make install && make seed && make start")
