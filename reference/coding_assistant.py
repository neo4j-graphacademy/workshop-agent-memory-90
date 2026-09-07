#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "neo4j-agent-memory[openai]==0.5.0",
#   "openai-agents",
#   "python-dotenv",
#   "httpx",
# ]
# ///
"""A small standalone coding assistant, backed by neo4j-agent-memory - the
same package and the same Aura database this repo's own generated app
(my-context-graph/) already uses, wired up without any of that app's own
scaffolding.

Run with `uv run reference/coding_assistant.py` from anywhere - the block
above tells uv what to install, so there is no venv to create or find, and
no reason to nest this inside another project's own environment.

This is a pattern to build on, not a finished assistant. Twelve things it
demonstrates, each added because an earlier version of this agent got them
wrong in practice:

- Instructions are split the same way memory_agent_mvp.py's own prompt is
  split, not rebuilt whole from scratch every turn: BASE_INSTRUCTIONS is
  static, built once, the same role openai-agents' single instructions
  slot gives memory_agent_mvp.py's static `system_prompt=SYSTEM_PROMPT`.
  dynamic_instructions is the addendum, playing the same role as
  memory_agent_mvp.py's `@agent.system_prompt async def
  what_you_remember` - it computes only what's genuinely turn-dependent
  (what the agent remembers, the tools it actually has, any pending
  self-edit) and appends it, so it never has to guess at either and the
  static rules aren't re-rendered on every turn for no reason.
- save_fact / save_preference / add_entity give it a real, deliberate way
  to write to the graph. Without these, "remember this" has no honest tool
  to reach for - and the dishonest alternative is inventing one.
- inspect_dependency reads an already-installed package's own metadata, no
  network required - check what you already have before reaching for a new
  dependency for a capability you may already have for free.
- Changing its own source goes through propose_self_edit, which only
  stages a proposal - applying it is not left to the model to remember.
  The REPL loop itself applies a staged edit, in plain code, the moment
  the user's very next reply is a plain confirmation, before the model is
  asked anything. A version that instead asked the model to call
  apply_self_edit once the user confirmed said "Confirmed." and skipped
  the call, the first time that was tried for real - a free-text
  confirmation step trusted to the model is not reliable enough for
  something this important.
- add_entity reads `entity.name` back off what add_entity() returns, not
  `entity.display_name` - a plausible-sounding field that doesn't exist on
  the real model. Verify a returned object's actual fields the same way
  (a pydantic model's own `.model_fields`, or plain `dir()`) before writing
  code that reads one back, the same way inspect_dependency checks a
  package before depending on it.
- Every turn passes Runner.run_streamed a real conversation history, read
  straight back from memory.short_term.get_conversation - not a bare
  string. `input` is genuinely all the model sees; passed only the current
  message, a follow-up like "now actually do what I asked" has nothing to
  resolve "what I asked" against, because the previous turn was never in
  the input at all. dynamic_instructions' own get_context() is a separate,
  semantic-similarity summary of longer-term memory - useful, but not a
  substitute for the actual prior turns, and too vague a query to reliably
  retrieve them.
- _tool_label checks a raw tool-call item's own `.type` discriminator
  ("web_search_call", "file_search_call") before falling back to the
  item's generic class name. ToolCallItem.tool_name reads `.name` off the
  raw item - which function-tool calls have, but WebSearchTool's raw item
  does not, confirmed by reading the SDK's own turn resolution code, not
  assumed. Without this, a real web search prints as the meaningless
  "[tool_call_item]", indistinguishable from any other tool call - which
  is exactly how a real web search got reported as one that never happened.
- SESSION_ID is generated fresh every process start, not a fixed constant.
  A version pinned to one constant string kept appending to and replaying
  the same never-ending conversation on every run - _conversation_input
  hands Runner.run_streamed real, raw prior turns, not just the current
  message, so an old, unrelated exchange from a previous day's run got
  replayed as if it had just been said, and the model answered that
  instead of the live question. Confirmed against the actual stored
  messages, not assumed: 69 messages turned up under one fixed session_id
  after only three real turns. Long-term memory (facts, preferences,
  reasoning traces) is untouched by this - get_context() surfaces those
  regardless of session_id - so restarting still remembers what should
  persist; only the raw-turn replay needed scoping to a single run.
- One turn's record-keeping - store the message, open the trace, try the
  model, close the trace on either outcome - lives inline in main()'s loop,
  the same way memory_agent_mvp.py's own main() does it, not factored into
  a separately callable function. A separate function is a second call
  site waiting to happen, and a second call site is exactly how this broke
  once already: a derived version of this file pulled the same logic into
  a standalone stream_turn(), added a --once code path that called it
  without the loop's try/except around it, and a failed turn crashed with
  a raw traceback instead of closing out on the record. Keeping it inline
  makes that specific mistake structurally impossible to make here - there
  is only one place a turn is ever processed, so there is nothing for a
  second call site to get wrong.
- trace_history reads this session's own reasoning traces back -
  report_step and complete_trace only ever write them. Without it, this
  agent has no way to look back at what it actually tried, proposed, or
  broke on past turns - which matters once the point isn't just building
  this agent, but later asking it to distil everything it learned about
  being edited into a skill of its own.
- Runner.run_streamed passes max_turns=40, not the SDK's own default of 10
  (agents/run_config.py: DEFAULT_MAX_TURNS = 10 - one turn per model
  round-trip). A task that reads and writes several files in sequence -
  authoring an AIP skill, for instance - needs more round-trips than that,
  and hit "Max turns (10) exceeded" mid-task on a real run, confirmed
  against the installed SDK's own source, not assumed.
- repository_command runs git/pytest/ruff/mypy/pyright/python/python3/uv
  against this repo - including `uv run path/to/script.py`, how AIP's own
  validate.py actually gets run. A first version's denylist forbade "run"
  outright and called itself read-only in its own docstring - neither true
  (a script it runs has write_file's own access, nothing more or less) nor
  useful (blocking "run" stopped nothing, since `python <file>` was never
  blocked and offers the same capability). install_python_package handles
  an explicitly requested pip install separately, so repository_command
  itself never needs to accept install/publish/push/commit/reset/clean/
  checkout/switch/merge/rebase or an inline one-liner (-c) - each of those
  either mutates state a tool call shouldn't silently commit to, or leaves
  nothing on disk for the user to review afterward.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

from agents import Agent, Runner, RunContextWrapper, WebSearchTool, function_tool
from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, MemoryClient, MemorySettings, Neo4jConfig
from neo4j_agent_memory.memory.reasoning import ToolCallStatus
from neo4j_agent_memory.memory.short_term import MessageRole
from pydantic import SecretStr

# Hosted-tool raw items carry no .name - only this type discriminator.
# See _tool_label.
_HOSTED_TOOL_LABELS = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
OWN_PATH = Path(__file__).resolve()
load_dotenv(REPO_ROOT / ".env")

# Fresh every process start - never a fixed constant. Runner.run_streamed
# below is handed real, raw conversation history (see _conversation_input),
# not just the current message, so anything stored under this session_id
# gets replayed to the model as if it were said moments ago. A constant
# session_id would mean every future run - months of separate, unrelated
# chats - keeps appending to and replaying the same never-ending thread.
# Long-term memory (facts, preferences, reasoning traces) is unaffected by
# this - dynamic_instructions' get_context() surfaces those regardless of
# session_id, so restarting still remembers what actually needs to persist;
# only the raw-turn replay is scoped to this run.
SESSION_ID = f"reference-agent-{uuid4().hex[:8]}"

# short_term.add_message() defaults to generate_embedding=True and embeds
# the message whole, with no length check of its own - text-embedding-3-small
# (MEMORY_EMBEDDING) caps input at 8192 tokens, and add_message awaits the
# embed() call inline, so a message over that limit raises straight out of
# add_message and kills the turn. A long user prompt or a reply quoting a
# large chunk of code is enough to cross it. 16,000 characters is a
# conservative margin under 8192 tokens even for token-dense text like code.
# The message itself is still stored either way - only the embedding (and
# therefore semantic recall of that one message) is skipped.
_MAX_EMBEDDABLE_CHARS = 16_000

# Set by propose_self_edit, consumed and cleared by _apply_staged_edit. A
# module-level global is enough here - this process handles one
# conversation at a time, so there is only ever one edit in flight.
_pending_self_edit: dict[str, str] | None = None

# Words that count as confirming a staged self-edit, checked by the REPL
# loop itself - not left to the model to recognise and act on.
CONFIRM_WORDS = {"confirm", "confirmed", "yes", "y", "ok", "okay", "apply", "apply it", "do it"}

# repository_command's allowlist - inspection, verification, and running a
# repo-local script, nothing that mutates git or dependency state. See
# repository_command itself for what's blocked and why.
_SAFE_COMMANDS = {"git", "pytest", "python", "python3", "uv", "ruff", "mypy", "pyright"}


def build_memory_settings() -> MemorySettings:
    """Read straight from this repo's own .env - the same one
    .vscode/mcp.json already points the memory MCP server at. No
    app-specific settings object in between, so this runs the same
    wherever it's invoked from, unlike a settings object built for a
    specific app's own boot sequence."""
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Missing required .env values: {', '.join(missing)}")
    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=SecretStr(os.environ["NEO4J_PASSWORD"]),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        ),
        embedding=os.environ["MEMORY_EMBEDDING"],
        # Same as this workshop's own generated app - see configure-app.py.
        extraction=ExtractionConfig(enable_gliner=False, enable_spacy=False),
    )


class TurnContext:
    """One of these per turn: the open memory client, this turn's raw
    message (so the dynamic instructions below can recall what's relevant
    to it), and the trace this turn is recording into. Tools reach it
    through RunContextWrapper - the SDK's own dependency-injection point -
    not a module-level global."""

    def __init__(self, memory: MemoryClient, trace_id: UUID, message: str, remembered: str):
        self.memory = memory
        self.trace_id = trace_id
        self.message = message
        # Computed once, before the run starts - dynamic_instructions reads
        # this instead of calling memory.get_context() itself, which the
        # SDK would otherwise re-run on every model round-trip within the
        # same turn (it rebuilds instructions before every step, not once
        # per user message), multiplying embedding and Neo4j round-trips by
        # however many tool calls a broad request chains together.
        self.remembered = remembered


async def report_step(
    ctx: RunContextWrapper[TurnContext],
    tool_name: str,
    arguments: dict,
    result: str,
    *,
    error: str | None = None,
) -> None:
    """Every tool call ends by reporting into the open trace - not a
    judgment call for the model to make, just something every tool does."""
    step = await ctx.context.memory.reasoning.add_step(
        ctx.context.trace_id,
        thought=f"Use {tool_name} to answer.",
        action=tool_name,
        observation=result[:2000],
        generate_embedding=False,
    )
    await ctx.context.memory.reasoning.record_tool_call(
        step.id,
        tool_name,
        arguments,
        result=result,
        status=ToolCallStatus.FAILURE if error else ToolCallStatus.SUCCESS,
        error=error,
    )


def _repo_path(relative_path: str) -> Path:
    """Every file tool goes through this - resolve, then confirm the
    result is still inside REPO_ROOT, so a path like `../../.env` can't
    reach outside this repository."""
    candidate = (REPO_ROOT / relative_path).resolve()
    candidate.relative_to(REPO_ROOT)  # raises ValueError if it escaped
    return candidate


@function_tool
async def read_file(ctx: RunContextWrapper[TurnContext], path: str) -> str:
    """Read a UTF-8 text file using a repository-relative path.

    Add further tools the same way - same decorator, same
    `ctx: RunContextWrapper[TurnContext]` first argument, same call to
    report_step before returning. A tool that lists files is a natural
    next one, so the agent can find a path before reading or writing it.
    """
    try:
        file_path = _repo_path(path)
        content = file_path.read_text(encoding="utf-8")
        result = content if len(content) <= 4000 else content[:4000] + f"\n(truncated, {len(content)} characters total)"
        await report_step(ctx, "read_file", {"path": path}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "read_file", {"path": path}, error, error=error)
        return f"Error: {error}"


@function_tool
async def write_file(ctx: RunContextWrapper[TurnContext], path: str, content: str) -> str:
    """Write a UTF-8 text file using a repository-relative path, creating
    parent directories if needed. Refuses to target this agent's own
    source file - use propose_self_edit and apply_self_edit for that
    instead, so a self-change is staged and confirmed before it lands."""
    try:
        file_path = _repo_path(path)
        if file_path == OWN_PATH:
            raise ValueError(
                "write_file does not touch this agent's own source - use "
                "propose_self_edit, then apply_self_edit once confirmed"
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        result = f"Wrote {len(content)} characters to {path}"
        await report_step(ctx, "write_file", {"path": path}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "write_file", {"path": path}, error, error=error)
        return f"Error: {error}"


@function_tool
async def save_fact(ctx: RunContextWrapper[TurnContext], subject: str, predicate: str, obj: str) -> str:
    """Record a durable fact as a subject/predicate/object triple, e.g.
    subject='this repo', predicate='depends on', obj='neo4j-agent-memory'."""
    try:
        await ctx.context.memory.long_term.add_fact(subject, predicate, obj)
        result = f"Recorded: {subject} {predicate} {obj}"
        arguments = {"subject": subject, "predicate": predicate, "obj": obj}
        await report_step(ctx, "save_fact", arguments, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(
            ctx, "save_fact", {"subject": subject, "predicate": predicate, "obj": obj}, error, error=error
        )
        return f"Error: {error}"


@function_tool
async def save_preference(ctx: RunContextWrapper[TurnContext], category: str, preference: str) -> str:
    """Record how the user wants this agent to work, filed under a
    category (e.g. category='code style', preference='no inline comments')."""
    try:
        await ctx.context.memory.long_term.add_preference(category, preference)
        result = f"Saved: {category} - {preference}"
        await report_step(ctx, "save_preference", {"category": category, "preference": preference}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(
            ctx, "save_preference", {"category": category, "preference": preference}, error, error=error
        )
        return f"Error: {error}"


@function_tool
async def add_entity(ctx: RunContextWrapper[TurnContext], name: str, entity_type: str, description: str = "") -> str:
    """Create or update an entity in the knowledge graph. entity_type is
    POLE+O: PERSON, ORGANIZATION, LOCATION, EVENT, or OBJECT."""
    try:
        entity, _ = await ctx.context.memory.long_term.add_entity(name, entity_type, description=description or None)
        # entity.name, not entity.display_name - the latter reads plausibly
        # but doesn't exist on the real model. Verified against Entity's
        # own model_fields, not assumed from how the name is used elsewhere.
        result = f"Recorded entity: {entity.name} ({entity.type})"
        await report_step(ctx, "add_entity", {"name": name, "entity_type": entity_type}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "add_entity", {"name": name, "entity_type": entity_type}, error, error=error)
        return f"Error: {error}"


@function_tool
async def trace_history(ctx: RunContextWrapper[TurnContext], limit: int = 20) -> str:
    """Read back this session's own reasoning traces - one line per turn,
    each with its id, task, outcome, and whether it succeeded. A summary
    only: list_traces/get_session_traces deliberately return traces without
    their steps (the package's own docstring says so - "use get_trace for
    full details"). The id printed here is what trace_detail takes to read
    one trace's actual steps and tool calls."""
    try:
        traces = await ctx.context.memory.reasoning.get_session_traces(SESSION_ID, limit=limit)
        lines = [f"[{t.id}] [{'success' if t.success else 'failed'}] {t.task} -> {t.outcome}" for t in traces]
        result = "\n".join(lines) or "No past traces yet."
        await report_step(ctx, "trace_history", {"limit": str(limit)}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "trace_history", {"limit": str(limit)}, error, error=error)
        return f"Error: {error}"


@function_tool
async def trace_detail(ctx: RunContextWrapper[TurnContext], trace_id: str) -> str:
    """Read one reasoning trace back in full - every step, in order, and
    every tool call under each step: name, arguments, result, status, error.
    trace_history only ever gives a one-line task/outcome summary; this is
    the only way to see what a past turn actually tried, proposed, or broke,
    step by step - which matters once the task is compiling that history
    into a skill, not just building this agent. Get a trace_id from
    trace_history first."""
    try:
        trace = await ctx.context.memory.reasoning.get_trace(trace_id)
        if trace is None:
            raise ValueError(f"no trace found for id {trace_id}")
        lines = [f"Trace {trace.id} [{'success' if trace.success else 'failed'}]: {trace.task} -> {trace.outcome}"]
        for step in trace.steps:
            lines.append(f"  step {step.step_number}: {step.thought or '(no thought recorded)'}")
            for call in step.tool_calls:
                error_suffix = f" error: {call.error}" if call.error else ""
                lines.append(f"    {call.tool_name}({call.arguments}) -> {call.status.value}: {call.result}{error_suffix}")
        result = "\n".join(lines)
        await report_step(ctx, "trace_detail", {"trace_id": trace_id}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "trace_detail", {"trace_id": trace_id}, error, error=error)
        return f"Error: {error}"


@function_tool
async def inspect_dependency(ctx: RunContextWrapper[TurnContext], package_name: str) -> str:
    """Read an installed Python package's own description straight from its
    metadata - no network call, no guessing from how this repo happens to
    use it. Check this before proposing a new external dependency for a
    capability one already installed (neo4j-agent-memory, openai-agents,
    ...) might already provide."""
    try:
        meta = importlib.metadata.metadata(package_name)
        description = meta.get_payload() or meta.get("Summary", "")
        result = description[:4000] if description else "(no description available)"
        await report_step(ctx, "inspect_dependency", {"package_name": package_name}, result)
        return result
    except importlib.metadata.PackageNotFoundError:
        error = f"{package_name} is not installed"
        await report_step(ctx, "inspect_dependency", {"package_name": package_name}, error, error=error)
        return f"Error: {error}"
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "inspect_dependency", {"package_name": package_name}, error, error=error)
        return f"Error: {error}"


@function_tool
async def repository_command(ctx: RunContextWrapper[TurnContext], command: str, timeout_seconds: int = 60) -> str:
    """Run a repository command for inspection, verification, or executing a
    script in this repo - git, pytest, ruff, mypy, pyright, python, python3,
    or uv (e.g. `uv run path/to/script.py`). Not a sandbox: a script this
    runs has the same access write_file already grants. Blocks git/dependency
    mutation (install, add, remove, publish, push, commit, reset, clean,
    checkout, switch, merge, rebase) and inline one-liners (-c), which leave
    nothing on disk to review afterward."""
    try:
        args = command.split()
        if not args or args[0] not in _SAFE_COMMANDS:
            raise ValueError(f"command must start with one of: {', '.join(sorted(_SAFE_COMMANDS))}")
        # "run" is not forbidden: uv run <script> is this tool's normal way
        # to execute a repo-local script (e.g. a vendored skill's own
        # validator), and blocking it protected nothing - python <file> was
        # never blocked and offers the same capability either way.
        forbidden = {"install", "add", "remove", "publish", "push", "commit", "reset", "clean", "checkout", "switch", "merge", "rebase", "-c"}
        if any(arg in forbidden for arg in args):
            raise ValueError("command contains a disallowed modifying or arbitrary-execution argument")
        completed = await asyncio.to_thread(subprocess.run, args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        result = json.dumps({"returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}, indent=2)
        await report_step(ctx, "repository_command", {"command": command, "timeout_seconds": str(timeout_seconds)}, result, error=result if completed.returncode else None)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "repository_command", {"command": command, "timeout_seconds": str(timeout_seconds)}, error, error=error)
        return f"Error: {error}"


@function_tool
async def install_python_package(ctx: RunContextWrapper[TurnContext], package_specifiers: list[str], timeout_seconds: int = 300) -> str:
    """Install explicitly requested Python package specifiers with python -m pip install.

    Only package specifiers are accepted: no flags, shell syntax, paths, URLs,
    or requirements files. This tool exists for immediate, user-requested
    environment repair and records the exact installation command.
    """
    try:
        if not package_specifiers:
            raise ValueError("at least one package specifier is required")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?(?:[<>=!~]{1,2}[A-Za-z0-9*_.+-]+)?", specifier) for specifier in package_specifiers):
            raise ValueError("package specifiers may contain only a package name, optional extras, and an optional version constraint")
        args = [sys.executable, "-m", "pip", "install", *package_specifiers]
        completed = await asyncio.to_thread(subprocess.run, args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        result = json.dumps({"command": ["python", "-m", "pip", "install", *package_specifiers], "returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}, indent=2)
        await report_step(ctx, "install_python_package", {"package_specifiers": ", ".join(package_specifiers), "timeout_seconds": str(timeout_seconds)}, result, error=result if completed.returncode else None)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "install_python_package", {"package_specifiers": ", ".join(package_specifiers), "timeout_seconds": str(timeout_seconds)}, error, error=error)
        return f"Error: {error}"


@function_tool
async def propose_self_edit(ctx: RunContextWrapper[TurnContext], new_content: str, reason: str) -> str:
    """Stage a change to this agent's own source file. Writes nothing -
    stores the proposed content and returns a preview. Tell the user what's
    staged and ask them to confirm - you do not need to call apply_self_edit
    yourself afterward, the session applies a staged edit automatically as
    soon as the user's next reply is a plain confirmation."""
    global _pending_self_edit
    try:
        compile(new_content, str(OWN_PATH), "exec")
        _pending_self_edit = {"content": new_content, "reason": reason}
        result = (
            f"Proposed, not yet applied: {reason} ({len(new_content)} characters staged). "
            "Tell the user what changed and ask them to confirm - a plain "
            "'confirm'/'yes'/'ok' reply applies it, no further tool call needed."
        )
        await report_step(ctx, "propose_self_edit", {"reason": reason}, result)
        return result
    except SyntaxError as exc:
        error = f"new_content is not valid Python, nothing staged: {exc}"
        await report_step(ctx, "propose_self_edit", {"reason": reason}, error, error=error)
        return f"Error: {error}"
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "propose_self_edit", {"reason": reason}, error, error=error)
        return f"Error: {error}"


def _apply_staged_edit() -> str:
    """Write the staged edit to disk and clear it. Pure file mechanics, no
    reasoning-trace bookkeeping - the REPL loop and apply_self_edit each
    record the trace themselves, since they run in different trace
    contexts (no open turn trace exists yet when the REPL applies an edit
    directly, ahead of calling the model at all)."""
    global _pending_self_edit
    if _pending_self_edit is None:
        raise RuntimeError("nothing staged - call propose_self_edit first")
    staged = _pending_self_edit
    OWN_PATH.with_suffix(".py.bak").write_text(OWN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    OWN_PATH.write_text(staged["content"], encoding="utf-8")
    _pending_self_edit = None
    return f"Applied: {staged['reason']}"


@function_tool
async def apply_self_edit(ctx: RunContextWrapper[TurnContext]) -> str:
    """Write the currently staged proposal (from propose_self_edit) to this
    agent's own source file. Normally unnecessary in the REPL - it applies a
    staged edit itself the moment the user confirms, without calling this
    tool. This exists for a non-interactive run where confirmation and
    application have to happen within the same tool-calling turn."""
    try:
        result = _apply_staged_edit()
        await report_step(ctx, "apply_self_edit", {}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "apply_self_edit", {}, error, error=error)
        return f"Error: {error}"


async def apply_confirmed_self_edit(memory: MemoryClient) -> str:
    """Apply a staged self-edit from the REPL loop directly, in plain code -
    not a tool call the model has to remember to make. A free-text
    confirmation step trusted to the model was tried first and failed the
    first time it mattered: the model replied "Confirmed." without calling
    apply_self_edit. This gate can't be talked out of firing, because the
    model is never asked to fire it - the REPL checks for a confirmation
    word itself and applies the edit before the model sees the message."""
    trace = await memory.reasoning.start_trace(
        session_id=SESSION_ID, task="apply confirmed self-edit", generate_embedding=False,
    )
    try:
        result = _apply_staged_edit()
        step = await memory.reasoning.add_step(
            trace.id,
            thought="User confirmed the staged self-edit.",
            action="apply_self_edit",
            observation=result,
            generate_embedding=False,
        )
        await memory.reasoning.record_tool_call(
            step.id, "apply_self_edit", {"confirmation": "confirm"}, result=result, status=ToolCallStatus.SUCCESS,
        )
        await memory.reasoning.complete_trace(trace.id, outcome=result, success=True)
        return result
    except Exception as exc:
        error = str(exc)
        await memory.reasoning.complete_trace(trace.id, outcome=error, success=False)
        return f"Error: {error}"


# Static policy - built once, the same role memory_agent_mvp.py's own
# module-level SYSTEM_PROMPT plays for pydantic_ai's system_prompt=. Every
# sentence here is turn-independent; only dynamic_instructions below
# changes per turn. Mirrors memory_agent_mvp.py's SYSTEM_PROMPT sentence
# for sentence, adapted to this agent's own tools where the tools differ
# (this agent has no find_similar_attendees or record_connection analogue,
# so those two sentences have no counterpart here):
BASE_INSTRUCTIONS = """You are a careful, honest coding assistant for this repository.

When you conclude something durable about this repository, its
dependencies, or how it works, record it with save_fact as a subject,
predicate, and object. When the user states how they want you to work,
record it with save_preference under a category. When a distinct person,
project, tool, or component worth tracking on its own comes up, record it
with add_entity. When the user is only describing something or thinking
out loud - not asking you to do anything - acknowledge what they said and
do not reach for a tool uninvited. Save anything worth keeping.

If none of these tools cover what you're asked to do, say so plainly and
stop. Never write a file that describes a workflow instead of using a
tool, and never say something was saved, searched, or changed unless a
tool call actually did it this turn.

Before proposing a new external dependency for a capability, call
inspect_dependency on what's already installed first - it may already
provide it for free.

Before writing code that reads an attribute off an object a library call
returned, verify that attribute is real - a pydantic model's own
`.model_fields`, or plain `dir()` on an instance - rather than assuming a
plausible-sounding name.

repository_command runs repository inspection, verification, and repo-local
scripts: git status/diff/log, pytest, ruff, mypy, pyright, and
python/python3/uv against files in this repo (including `uv run
path/to/script.py`). It is not a sandbox - a script it runs has the same
access write_file already grants, nothing more, nothing less. It still
refuses anything that mutates git or dependency state (install, publish,
push, commit, reset, clean, checkout, switch, merge, rebase) and inline
one-liners (-c), which leave nothing on disk to review afterward. Use
install_python_package for an explicitly requested Python package
installation instead; it permits only `python -m pip install
<package-specifiers>` and does not accept shell syntax or arbitrary pip
flags.

Investigate in small steps. After one or two tool calls, report back what
you found rather than chaining many tool calls unprompted in the same
turn - each round-trip resends everything called so far this turn, and an
open-ended request answered in one long unattended chain is how a turn
outgrows the model's own context window before it can reply. Prefer
several short turns, checking in between each one, over a single very
long one."""


async def dynamic_instructions(ctx: RunContextWrapper[TurnContext], agent: Agent[TurnContext]) -> str:
    """The per-turn addendum to BASE_INSTRUCTIONS - plays the same role as
    memory_agent_mvp.py's `@agent.system_prompt async def
    what_you_remember`, which appends a fresh memory summary to a static
    system_prompt rather than rebuilding the whole prompt from scratch
    every turn. openai-agents gives an Agent one instructions slot, not
    pydantic_ai's composable multiple @agent.system_prompt functions, so
    the same static/dynamic split is done by hand here: BASE_INSTRUCTIONS
    above never changes, and only what's genuinely turn-dependent -
    remembered context, the agent's actual live tool list (through the
    agent parameter, passed in automatically), and any pending self-edit -
    is computed and appended here. `remembered` is read from the turn
    context, not recomputed here - openai-agents calls this function fresh
    before every model round-trip within a turn, not once per user
    message, so awaiting memory.get_context() in this function's own body
    would re-run four vector searches on every single tool-calling step."""
    remembered = ctx.context.remembered
    tool_lines = "\n".join(
        # WebSearchTool has no .description at all - confirmed against the
        # installed class. inspect.getdoc() falls back to the class's own
        # docstring ("A hosted tool that lets the LLM search the web...") for
        # exactly this case, instead of a placeholder that reads as "not a
        # real capability" and gets the tool reported as missing entirely.
        f"- {tool.name}: {getattr(tool, 'description', None) or inspect.getdoc(tool) or '(no description available)'}"
        for tool in agent.tools
    )
    pending_note = ""
    if _pending_self_edit is not None:
        pending_note = (
            f"\n\nA self-edit is staged but not applied yet: {_pending_self_edit['reason']}\n"
            "Tell the user what's staged and ask them to confirm. You do not "
            "need to call apply_self_edit yourself - the session applies a "
            "staged edit automatically as soon as the user's next reply is a "
            "plain confirmation (e.g. 'confirm', 'yes', 'ok')."
        )

    return f"""{BASE_INSTRUCTIONS}

What you remember from past turns:
{remembered or "(nothing yet)"}

Your tools, exactly as installed - nothing else:
{tool_lines}{pending_note}"""


agent = Agent[TurnContext](
    name="Reference coding assistant",
    instructions=dynamic_instructions,
    tools=[
        read_file,
        write_file,
        save_fact,
        save_preference,
        add_entity,
        trace_history,
        trace_detail,
        inspect_dependency,
        repository_command,
        install_python_package,
        propose_self_edit,
        apply_self_edit,
        WebSearchTool(),
    ],
)


def _tool_label(item: object) -> str:
    """Best-effort display label for a streamed tool-call item, using the
    SDK's own resolved name rather than a hardcoded map - see tool_name on
    ToolCallItem, which falls back to the raw payload's name itself. That
    works for a function tool, but WebSearchTool and file-search calls carry
    no .name at all - only a raw_item.type discriminator - so check
    _HOSTED_TOOL_LABELS before falling back to the item's own generic type."""
    name = getattr(item, "tool_name", None)
    if name:
        return str(name)
    raw_item = getattr(item, "raw_item", None)
    raw_type = getattr(raw_item, "type", None)
    if raw_type in _HOSTED_TOOL_LABELS:
        return _HOSTED_TOOL_LABELS[raw_type]
    title = getattr(item, "title", None)
    if title:
        return str(title)
    return type(item).__name__


async def _conversation_input(memory: MemoryClient, *, limit: int = 20) -> list[dict[str, str]]:
    """The actual recent turns, verbatim, read straight back from the same
    short-term store every turn already writes to - not a second history to
    keep in sync. Runner.run_streamed only ever sees what's passed as
    `input`; passed a bare string, every call starts from a blank slate, and
    the only thing standing in for the rest of the conversation is
    dynamic_instructions' get_context() - a semantic-similarity summary, not
    a transcript. A vague follow-up ("now actually do what I asked") barely
    matches anything, so that summary comes back near-empty and the model
    has no way to know what "what I asked" refers to - confirmed by reading
    Runner.run_streamed's own signature, which only accepts a single string
    or an explicit list of prior turns; it keeps no state between calls."""
    conversation = await memory.short_term.get_conversation(SESSION_ID, limit=limit)
    return [
        {"role": "user" if item.role == MessageRole.USER else "assistant", "content": item.content}
        for item in conversation.messages
        if item.role in (MessageRole.USER, MessageRole.ASSISTANT)
    ]


async def main() -> None:
    settings = build_memory_settings()
    async with MemoryClient(settings) as memory:
        print(f"Coding assistant ready (session {SESSION_ID}). Type /help for commands or /exit to quit.\n")
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                break
            if user_input == "/help":
                print("/help  show this message\n/exit  close the session")
                continue
            if _pending_self_edit is not None and user_input.strip().lower() in CONFIRM_WORDS:
                print(f"agent> {await apply_confirmed_self_edit(memory)}\n")
                continue

            # Store the message first, so even a failed turn leaves a
            # complete record - and everything below, inline in this loop,
            # the same way memory_agent_mvp.py's own main() does it. See
            # this file's docstring: a separate run_turn() function used to
            # live here, and factoring it out is exactly what let a derived
            # version of this file grow a second, unsafe call site.
            await memory.short_term.add_message(
                SESSION_ID, "user", user_input, generate_embedding=len(user_input) <= _MAX_EMBEDDABLE_CHARS,
            )

            # Open the trace before the attempt - either ending closes it.
            trace = await memory.reasoning.start_trace(
                session_id=SESSION_ID, task=user_input[:200], generate_embedding=False,
            )
            remembered = await memory.get_context(user_input, session_id=SESSION_ID, max_items=5)
            turn = TurnContext(memory=memory, trace_id=trace.id, message=user_input, remembered=remembered)
            conversation_input = await _conversation_input(memory)

            print("agent> ", end="", flush=True)
            response_parts: list[str] = []
            try:
                # openai-agents defaults to max_turns=10 (one turn per
                # model round-trip) - too low for a task that reads and
                # writes several files in sequence, e.g. authoring an AIP
                # skill. Confirmed against a real run that hit "Max turns
                # (10) exceeded" mid-task. Raised to 20, not higher - every
                # round-trip resends everything called so far this turn,
                # tool outputs included, so a broad request left unattended
                # for 40 rounds can outgrow the model's own context window
                # before it ever replies. 20 still covers a real multi-file
                # task; it hits a clear, recoverable "Max turns exceeded"
                # instead.
                result = Runner.run_streamed(agent, conversation_input, context=turn, max_turns=20)
                async for event in result.stream_events():
                    if event.type == "raw_response_event":
                        delta = getattr(event.data, "delta", None)
                        if delta:
                            response_parts.append(delta)
                            print(delta, end="", flush=True)
                    elif event.type == "run_item_stream_event" and event.name == "tool_called":
                        print(f"\n[{_tool_label(event.item)}]", end="", flush=True)
                print()
                response = "".join(response_parts)
                await memory.short_term.add_message(
                    SESSION_ID, "assistant", response, generate_embedding=len(response) <= _MAX_EMBEDDABLE_CHARS,
                )
                await memory.reasoning.complete_trace(trace.id, outcome=response[:500], success=True)
            except Exception as exc:
                await memory.reasoning.complete_trace(trace.id, outcome=str(exc), success=False)
                print(f"\nerror> {exc}")

        print("\nGoodbye. Everything you told me is saved - run me again and I will remember.")


if __name__ == "__main__":
    asyncio.run(main())
