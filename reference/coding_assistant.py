#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "neo4j-agent-memory[openai]==0.5.0",
#   "neo4j-graphrag[openai]>=1.19.0",
#   "openai-agents==0.3.3",
#   "python-dotenv",
#   "httpx",
# ]
# ///
"""A small standalone coding assistant, backed by neo4j-agent-memory - the
same package and the same Aura database this repo's own generated app
(my-context-graph/) already uses, wired up without any of that app's own
scaffolding.

Run with `uv run reference/coding_assistant.py` from anywhere - the block
above tells uv what to install, so there is no venv to create or find.

Eleven tools:

- read and write over this repository. write refuses this file's own
  source - propose_self_edit and apply_self_edit are the only path to
  change it.
- propose_self_edit stages an old_string -> new_string patch (exact
  match, unique in the file) and only stages it, building on whatever is
  already staged rather than the unmodified file, so a multi-part change
  (a new function plus its entry in the tools list) is more than one call
  in the same turn. The REPL applies a staged edit itself, in plain code,
  the moment the user's next reply is a plain confirmation - never in the
  same turn it was proposed, never left to the model to remember to do,
  and the fact of what changed is recorded automatically at that point
  too, since that is the one turn the model never runs on.
- save_fact / save_preference / add_entity / recall / resume: durable
  memory, both directions. Without the first three, "remember this" has
  no honest tool to reach for. recall searches facts, preferences,
  entities, reasoning traces, and past messages by meaning, across every
  session. resume is different: SESSION_ID is fresh every process start,
  so the live conversation history is empty on restart even though a
  previous run's messages are still in the graph - resume finds the most
  recent other session by recency and returns its messages verbatim,
  which a query-based search over facts alone cannot reliably do.
- get_schema and query_database: neo4j-agent-memory is not a graph
  querying tool - its API performs only the fixed operations it was built
  for. These use neo4j_graphrag's own Text2CypherRetriever, straight over
  the same database, for the questions that actually need Cypher: schema
  introspection and structured, computed answers recall's semantic search
  can't give. A second driver, separate from MemoryClient's internal one -
  see build_graphrag_driver's own docstring for why.
- add_entity reads `entity.name` back off what add_entity() returns, not
  `entity.display_name` - a plausible-sounding field that doesn't exist on
  the real model. Verify a returned object's actual fields (a pydantic
  model's own `.model_fields`, or plain `dir()`) before writing code that
  reads one back.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

from agents import Agent, Runner, RunContextWrapper, function_tool
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j_agent_memory import ExtractionConfig, MemoryClient, MemorySettings, Neo4jConfig
from neo4j_agent_memory.memory.reasoning import ToolCallStatus
from neo4j_agent_memory.memory.short_term import MessageRole
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from pydantic import SecretStr

# The driver logs every server notification (e.g. a deprecated procedure)
# through these two loggers by default - silenced so a normal tool call
# doesn't print a WARNING block into the middle of the REPL's own output.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j.session").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
OWN_PATH = Path(__file__).resolve()
load_dotenv(REPO_ROOT / ".env")

# Fresh every process start, not a fixed constant. Long-term memory and
# reasoning traces search across every session regardless; only the raw
# conversation replay below (_conversation_input) is scoped to one run.
SESSION_ID = f"reference-agent-{uuid4().hex[:8]}"

# short_term.add_message() defaults to generate_embedding=True and embeds
# the message whole, with no length check of its own - text-embedding-3-small
# (MEMORY_EMBEDDING) caps input at 8192 tokens, and add_message awaits the
# embed() call inline, so a message over that limit raises straight out of
# add_message and kills the turn. 16,000 characters is a conservative margin
# even for token-dense text like code. The message is still stored either
# way - only its embedding (and therefore semantic recall of it) is skipped.
_MAX_EMBEDDABLE_CHARS = 16_000

# Read by default, no truncation - see read()'s own docstring for when the
# cap below actually applies.
_MAX_READ_CHARS = 100_000

# Set by propose_self_edit, consumed and cleared by _apply_staged_edit. A
# module-level global is enough here - this process handles one
# conversation at a time, so there is only ever one edit in flight.
_pending_self_edit: dict[str, str] | None = None

# Words that count as confirming a staged self-edit, checked by the REPL
# loop itself - not left to the model to recognise and act on.
CONFIRM_WORDS = {"confirm", "confirmed", "yes", "y", "ok", "okay", "apply", "apply it", "do it"}


DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def build_memory_settings() -> MemorySettings:
    """Read straight from this repo's own .env - the same one
    .vscode/mcp.json already points the memory MCP server at."""
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Missing required .env values: {', '.join(missing)}")
    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=SecretStr(os.environ["NEO4J_PASSWORD"]),
            database=DATABASE,
        ),
        embedding=os.environ["MEMORY_EMBEDDING"],
        # Same as this workshop's own generated app - see configure-app.py.
        extraction=ExtractionConfig(enable_gliner=False, enable_spacy=False),
    )


def build_graphrag_driver() -> Driver:
    """A second, raw driver onto the same Neo4j instance MemoryClient
    already connects to (same env vars, same database) - not a different
    graph. neo4j-agent-memory's own API only performs the fixed operations
    it was built for (add a fact, search entities by meaning); it has no
    way to answer an arbitrary question about the graph's actual current
    shape. get_schema and query_database need direct query execution for
    that, so they get their own driver rather than reaching into
    MemoryClient's internal one, which isn't part of its public surface.
    Built once in main() and closed there - not at import time, since a
    bare import (for testing this file's own tools, for instance) should
    not require a live Neo4j connection to succeed."""
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


class TurnContext:
    """One of these per turn: the open memory client, the trace this turn
    is recording into, a first-pass guess at what's relevant to it (see
    main() - recall() is the retriable version of that same guess), and
    the raw driver/retriever get_schema and query_database use - built
    once in main(), unchanged turn to turn, just threaded through the same
    way memory already is. Tools reach it through RunContextWrapper - the
    SDK's own dependency-injection point - not a module-level global."""

    def __init__(
        self,
        memory: MemoryClient,
        trace_id: UUID,
        remembered: str,
        driver: Driver,
        text2cypher: Text2CypherRetriever,
    ):
        self.memory = memory
        self.trace_id = trace_id
        self.remembered = remembered
        self.driver = driver
        self.text2cypher = text2cypher


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
        step.id, tool_name, arguments, result=result,
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
async def read(ctx: RunContextWrapper[TurnContext], path: str) -> str:
    """Read a repository-relative text file, in full. Only truncates past
    _MAX_READ_CHARS, which is large enough that this should never matter
    for source files - it exists for the rare case of pointing this at a
    huge log or data file by mistake."""
    try:
        content = _repo_path(path).read_text(encoding="utf-8")
        result = (
            content
            if len(content) <= _MAX_READ_CHARS
            else content[:_MAX_READ_CHARS] + f"\n(truncated, {len(content)} characters total)"
        )
        await report_step(ctx, "read", {"path": path}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "read", {"path": path}, error, error=error)
        return f"Error: {error}"


@function_tool
async def write(
    ctx: RunContextWrapper[TurnContext], path: str, content: str, old_string: str | None = None
) -> str:
    """Write a repository-relative file. Refuses this agent's own source -
    use propose_self_edit, then apply_self_edit once confirmed, so a
    self-change is staged and confirmed before it lands.

    Two modes, chosen by whether old_string is given:
    - Whole file (old_string omitted): `content` becomes the entire file.
      For a new file, or a genuine full replacement.
    - Patch (old_string given): `content` replaces old_string, which must
      match the file's current text exactly once - include a line or two
      of surrounding context if the snippet isn't unique by itself. Use
      this for any change to an existing file: it costs the size of the
      change, not the size of the file.

    A .py target is compiled before anything is written, and a change to
    an existing file is backed up to <path>.bak first.
    """
    try:
        file_path = _repo_path(path)
        if file_path == OWN_PATH:
            raise ValueError(
                "write does not touch this agent's own source - use "
                "propose_self_edit, then apply_self_edit once confirmed"
            )
        if old_string is None:
            new_content = content
        else:
            if not file_path.exists():
                raise ValueError("old_string given but the file doesn't exist yet - omit it to create the file")
            current = file_path.read_text(encoding="utf-8")
            occurrences = current.count(old_string)
            if occurrences == 0:
                raise ValueError("old_string not found - re-read the file, it may have changed")
            if occurrences > 1:
                raise ValueError(f"old_string matches {occurrences} times - add more context so it matches once")
            new_content = current.replace(old_string, content, 1)

        if file_path.suffix == ".py":
            compile(new_content, str(file_path), "exec")

        if file_path.exists():
            file_path.with_suffix(file_path.suffix + ".bak").write_text(
                file_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_content, encoding="utf-8")

        result = f"Wrote {len(new_content)} characters to {path}"
        await report_step(ctx, "write", {"path": path, "old_string": old_string}, result)
        return result
    except SyntaxError as exc:
        error = f"resulting file is not valid Python, nothing written: {exc}"
        await report_step(ctx, "write", {"path": path, "old_string": old_string}, error, error=error)
        return f"Error: {error}"
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "write", {"path": path, "old_string": old_string}, error, error=error)
        return f"Error: {error}"


@function_tool
async def propose_self_edit(
    ctx: RunContextWrapper[TurnContext], old_string: str, new_string: str, reason: str
) -> str:
    """Stage a change to this agent's own source file as an exact,
    uniquely-matching old_string -> new_string replacement - never the whole
    file. Writes nothing yet - stores the resulting content and returns a
    preview. Tell the user what's staged and ask them to confirm - you do
    not need to call apply_self_edit yourself afterward, the session applies
    a staged edit automatically as soon as the user's next reply is a plain
    confirmation.

    Call this more than once in the same turn if one patch isn't the whole
    change - each call builds on whatever is already staged, not on the
    unmodified file, so adding a new tool is normally two or three calls:
    one to insert the function, one to add its name to the tools list, and
    (if needed) one for the docstring. Only ask for confirmation once the
    staged result is a complete, working change - a docs-only edit that
    describes a tool without the function that implements it, or a tool
    that exists but isn't registered in the tools list, is not complete.

    old_string must match the current staged source exactly once. Include a
    line or two of surrounding context if the snippet you want to change
    isn't unique by itself - read it first if you're not sure it still
    matches, since a prior edit (yours, in this same turn, or the user's)
    may have moved it."""
    global _pending_self_edit
    try:
        current = _pending_self_edit["content"] if _pending_self_edit is not None else OWN_PATH.read_text(encoding="utf-8")
        occurrences = current.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string not found in the current source - re-read the file, it may have changed")
        if occurrences > 1:
            raise ValueError(
                f"old_string matches {occurrences} times - include more surrounding context so it matches exactly once"
            )
        new_content = current.replace(old_string, new_string, 1)
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
        error = f"resulting file is not valid Python, nothing staged: {exc}"
        await report_step(ctx, "propose_self_edit", {"reason": reason}, error, error=error)
        return f"Error: {error}"
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "propose_self_edit", {"reason": reason}, error, error=error)
        return f"Error: {error}"


async def _apply_staged_edit(memory: MemoryClient) -> str:
    """Write the staged edit to disk, clear it, and record what changed as
    a durable fact - in code, not left to the model to remember. Both
    apply paths (the REPL's own confirm handling and the apply_self_edit
    tool) go through this one function; recording the fact here, not in
    each caller, is why it can't be skipped by either. Without this, the
    only record of a self-edit was whatever the model said while proposing
    it - and the REPL applies a confirmed edit without the model ever
    running on that turn, so nothing here would otherwise get recorded at
    the one moment the change actually lands. recall() searches every
    fact in the shared graph with no sense of recency - a self-edit this
    process never recorded is indistinguishable from one that happened in
    an entirely different session, and a stale unrelated fact wins by
    default when nothing current exists to compete with it."""
    global _pending_self_edit
    if _pending_self_edit is None:
        raise RuntimeError("nothing staged - call propose_self_edit first")
    staged = _pending_self_edit
    OWN_PATH.with_suffix(".py.bak").write_text(OWN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    OWN_PATH.write_text(staged["content"], encoding="utf-8")
    _pending_self_edit = None
    result = f"Applied: {staged['reason']}"
    await memory.long_term.add_fact("this agent's own source", "was changed to", staged["reason"])
    return result


@function_tool
async def apply_self_edit(ctx: RunContextWrapper[TurnContext]) -> str:
    """Write the currently staged proposal (from propose_self_edit) to this
    agent's own source file. Normally unnecessary in the REPL - it applies a
    staged edit itself the moment the user confirms, without calling this
    tool. This exists for a non-interactive run where confirmation and
    application have to happen within the same tool-calling turn."""
    try:
        result = await _apply_staged_edit(ctx.context.memory)
        await report_step(ctx, "apply_self_edit", {}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "apply_self_edit", {}, error, error=error)
        return f"Error: {error}"


async def apply_confirmed_self_edit(memory: MemoryClient) -> str:
    """Apply a staged self-edit from the REPL loop directly, in plain code -
    not a tool call the model has to remember to make. The REPL checks for
    a confirmation word itself and applies the edit before the model ever
    sees the message, so this can't be talked out of firing."""
    trace = await memory.reasoning.start_trace(
        session_id=SESSION_ID, task="apply confirmed self-edit", generate_embedding=False,
    )
    try:
        result = await _apply_staged_edit(memory)
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
        await report_step(ctx, "save_fact", {"subject": subject, "predicate": predicate, "obj": obj}, error, error=error)
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
        await report_step(ctx, "save_preference", {"category": category, "preference": preference}, error, error=error)
        return f"Error: {error}"


@function_tool
async def add_entity(ctx: RunContextWrapper[TurnContext], name: str, entity_type: str, description: str = "") -> str:
    """Create or update an entity in the knowledge graph. entity_type is
    POLE+O: PERSON, ORGANIZATION, LOCATION, EVENT, or OBJECT."""
    try:
        entity, _ = await ctx.context.memory.long_term.add_entity(name, entity_type, description=description or None)
        # entity.name, not entity.display_name - the latter reads plausibly
        # but doesn't exist on the real model. Verified against Entity's
        # own model_fields, not assumed.
        result = f"Recorded entity: {entity.name} ({entity.type})"
        await report_step(ctx, "add_entity", {"name": name, "entity_type": entity_type}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "add_entity", {"name": name, "entity_type": entity_type}, error, error=error)
        return f"Error: {error}"


@function_tool
async def recall(ctx: RunContextWrapper[TurnContext], query: str) -> str:
    """Search memory - facts, preferences, entities, past reasoning traces,
    and past conversation messages - by meaning, across every session, not
    just this one. Call this whenever what you already remember (below, in
    your own instructions) isn't enough, with whatever query is actually
    relevant - it can be called more than once with a refined query if the
    first result isn't useful.

    Facts alone are not enough to answer "what did we do last time":
    nothing writes a fact unless something (a self-edit landing, an
    explicit save_fact call) is deliberate about it, so a real prior
    conversation can be entirely unrepresented in facts while still being
    fully present as messages. include_short_term stays on for exactly
    this reason - the current turn's own recent history is already in the
    conversation input Runner.run_streamed was given, but a search across
    every OTHER session's messages is not, and this is the only thing that
    surfaces it."""
    try:
        result = await ctx.context.memory.get_context(query, max_items=8)
        result = result or "Nothing matched that query."
        await report_step(ctx, "recall", {"query": query}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "recall", {"query": query}, error, error=error)
        return f"Error: {error}"


@function_tool
async def resume(ctx: RunContextWrapper[TurnContext]) -> str:
    """Read back the most recent OTHER session's conversation, verbatim -
    not a semantic guess, the actual messages in order. SESSION_ID is fresh
    every process start, so restarting this agent (after a self-edit, or
    just closing and reopening it) means the live conversation history
    Runner.run_streamed sees is empty, even though everything from before
    is still sitting in the graph under a different session_id. Call this
    when the user asks to pick up where a previous run left off, or asks
    what happened last time and recall's semantic search doesn't turn up
    an answer specific enough to be useful."""
    try:
        sessions = await ctx.context.memory.short_term.list_sessions(
            limit=5, order_by="updated_at", order_dir="desc"
        )
        previous = next((s for s in sessions if s.session_id != SESSION_ID), None)
        if previous is None:
            result = "No prior session found - this is the first run."
        else:
            conversation = await ctx.context.memory.short_term.get_conversation(previous.session_id, limit=20)
            lines = [f"[{m.role.value}] {m.content}" for m in conversation.messages]
            result = f"Last session ({previous.session_id}, updated {previous.updated_at}):\n" + "\n".join(lines)
        await report_step(ctx, "resume", {}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "resume", {}, error, error=error)
        return f"Error: {error}"


@function_tool
async def get_schema(ctx: RunContextWrapper[TurnContext]) -> str:
    """Read the graph's actual current node labels and relationship types
    straight from the database. Call this first if you're unsure how the
    graph is structured, before writing a Cypher-shaped question for
    query_database - neo4j-agent-memory's own tools only ever perform the
    fixed operations they were built for (add a fact, search entities by
    meaning); this one can look at the graph's real, current shape,
    whatever it happens to be."""
    try:
        records, _, _ = ctx.context.driver.execute_query(
            "CALL db.schema.visualization()", database_=DATABASE,
        )
        result = json.dumps([record.data() for record in records], default=str)
        await report_step(ctx, "get_schema", {}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "get_schema", {}, error, error=error)
        return f"Error: {error}"


@function_tool
async def query_database(ctx: RunContextWrapper[TurnContext], query: str) -> str:
    """Answer a specific, structured question about this graph - "how many
    facts exist", "what entities has this agent recorded" - by converting
    it to Cypher and running it, via neo4j_graphrag's Text2CypherRetriever.
    Use this when the question needs an exact answer computed over the
    graph's real data, not a semantic guess at what's relevant - that's
    what recall is for. Call get_schema first if you don't already know
    the graph's actual labels and relationship types; a query built
    against the wrong shape returns nothing, not an error."""
    try:
        search_result = ctx.context.text2cypher.search(query_text=query)
        result = "\n".join(item.content for item in search_result.items) or "No results."
        await report_step(ctx, "query_database", {"query": query}, result)
        return result
    except Exception as exc:
        error = str(exc)
        await report_step(ctx, "query_database", {"query": query}, error, error=error)
        return f"Error: {error}"


BASE_INSTRUCTIONS = """You are a careful, honest coding assistant for this repository.

You have exactly eleven tools: read and write over this repository,
propose_self_edit and apply_self_edit to change this file's own source,
five memory tools - save_fact, save_preference, add_entity, recall, and
resume - and get_schema plus query_database for the graph itself. Nothing
else exists. If a task needs an action none of these eleven can ever
perform - running a shell command, calling a network API
- say so plainly and stop. A missing tool is different: propose_self_edit
exists precisely so you can write that tool's real implementation
yourself, right now. If you conclude a tool is missing, write it: a real
function, added to the tools list, following the same pattern as the
tools already here. That is normally more than one propose_self_edit
call - the function body and its entry in the tools list are two
different places in the file - and every call in the same turn builds on
whatever you already staged, so keep calling it until the staged result
actually works. Do not stop after only describing the tool in a docstring
or a comment, and do not ask for confirmation on a patch you know is
incomplete - a tool mentioned in prose but not implemented, or
implemented but not registered in the tools list, is not done. Do not
stage a no-op patch (old_string equal to new_string) either, and do not
defer to "an execution backend that isn't available yet" - writing the
tool is the execution backend. Never write a file that describes a
workflow instead of using a tool, and never say something was read,
written, saved, or recalled unless a tool call actually did it this turn.

write refuses your own source file. To change it: call propose_self_edit
with an old_string/new_string patch, tell the user what you're proposing
and why, then stop and wait - do not call apply_self_edit yourself. The
session applies a staged edit automatically the moment the user's next
reply is a plain confirmation ('confirm', 'yes', 'ok'). Never treat your
own proposal as applied until that happens.

When you conclude something durable about this repository, its
dependencies, or how it works, record it with save_fact as a subject,
predicate, and object. When the user states how they want you to work,
record it with save_preference under a category. When a distinct person,
project, tool, or component worth tracking on its own comes up, record it
with add_entity. When the user is only describing something or thinking
out loud - not asking you to do anything - acknowledge what they said and
do not reach for a tool uninvited. Save anything worth keeping.

Before answering from memory alone, call recall with a query specific to
what you actually need - what you already remember below is a single
best-effort guess made before you saw this message; recall lets you
retry it, refine it, or confirm nothing relevant exists. When the user
asks to continue, resume, or otherwise refers to a previous run of this
same agent - not just something mentioned earlier - call resume instead
of recall: this session's own SESSION_ID is brand new every process
start, so the conversation you can already see never includes an earlier
run, and resume finds the most recent one by recency, not by how well a
query happens to match it.

recall, get_schema, and query_database all touch the same graph but
answer different kinds of question. recall is a semantic guess at what's
relevant - it can miss, and it can't count or compute anything. When the
question has an exact, structured answer - how many of something exist,
which entities are connected, what the graph actually contains right now
- call get_schema first if you don't already know the labels and
relationship types, then query_database. Don't use query_database as a
second attempt at something recall already answered well; use it when
the question needed real graph structure, not meaning, in the first
place.

Before writing code that reads an attribute off an object a library call
returned, verify that attribute is real - a pydantic model's own
`.model_fields`, or plain `dir()` on an instance - rather than assuming a
plausible-sounding name.

write's old_string must match the file's current text exactly once. If
you're not sure it still matches, read the file first - a prior edit,
yours or the user's, may have moved it."""


async def dynamic_instructions(ctx: RunContextWrapper[TurnContext], agent: Agent[TurnContext]) -> str:
    """The per-turn addendum to BASE_INSTRUCTIONS: what's genuinely
    turn-dependent - remembered context, and the agent's actual live tool
    list - computed and appended here rather than baked into the static
    prompt above, which never changes."""
    remembered = ctx.context.remembered
    tool_lines = "\n".join(
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
        read, write, propose_self_edit, apply_self_edit,
        save_fact, save_preference, add_entity, recall, resume,
        get_schema, query_database,
    ],
)


def _tool_label(item: object) -> str:
    """The name to print next to a streamed tool call. ToolCallItem itself
    carries no tool_name field at all in the installed SDK (confirmed
    against agents.items.ToolCallItem's own source, not assumed) - only
    raw_item, the actual Responses API item, has the real name. A function
    call's raw_item.name is that name directly; a hosted tool (web search,
    file search) has no .name but does have a .type discriminator
    ("web_search_call") worth showing instead of nothing. Only fall back
    to the class name if neither is present."""
    raw_item = getattr(item, "raw_item", None)
    name = getattr(raw_item, "name", None)
    if name:
        return str(name)
    raw_type = getattr(raw_item, "type", None)
    if raw_type:
        return str(raw_type)
    return type(item).__name__


async def _conversation_input(memory: MemoryClient, *, limit: int = 20) -> list[dict[str, str]]:
    """The actual recent turns, verbatim, read straight back from the same
    short-term store every turn already writes to. Runner.run_streamed only
    ever sees what's passed as `input`; passed a bare string, every call
    starts from a blank slate, and a vague follow-up ("now actually do what
    I asked") has nothing to resolve "what I asked" against."""
    conversation = await memory.short_term.get_conversation(SESSION_ID, limit=limit)
    return [
        {"role": "user" if item.role == MessageRole.USER else "assistant", "content": item.content}
        for item in conversation.messages
        if item.role in (MessageRole.USER, MessageRole.ASSISTANT)
    ]


async def main() -> None:
    settings = build_memory_settings()
    # A second, raw driver for get_schema/query_database - see
    # build_graphrag_driver's own docstring for why this isn't built at
    # import time and doesn't reach into MemoryClient's internal one.
    driver = build_graphrag_driver()
    text2cypher = Text2CypherRetriever(
        driver=driver, llm=OpenAILLM(model_name="gpt-5.2"), neo4j_database=DATABASE,
    )
    try:
        async with MemoryClient(settings) as memory:
            print(f"Coding assistant ready (session {SESSION_ID}). Type /exit to quit.\n")
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
                if _pending_self_edit is not None and user_input.strip().lower() in CONFIRM_WORDS:
                    print(f"agent> {await apply_confirmed_self_edit(memory)}\n")
                    continue

                # Store the message first, so even a failed turn leaves a
                # complete record. Everything below stays inline in this loop,
                # not factored into a separate function - there is only one
                # place a turn is ever processed.
                await memory.short_term.add_message(
                    SESSION_ID, "user", user_input, generate_embedding=len(user_input) <= _MAX_EMBEDDABLE_CHARS,
                )

                # Open the trace before the attempt - either ending closes it.
                trace = await memory.reasoning.start_trace(
                    session_id=SESSION_ID, task=user_input[:200], generate_embedding=False,
                )
                # One best-effort guess at what's relevant, made before the model
                # sees the turn - recall() is the retriable version of this same
                # call, for when this guess isn't enough.
                remembered = await memory.get_context(user_input, session_id=SESSION_ID, max_items=5)
                turn_context = TurnContext(
                    memory=memory, trace_id=trace.id, remembered=remembered,
                    driver=driver, text2cypher=text2cypher,
                )
                conversation_input = await _conversation_input(memory)

                print("agent> ", end="", flush=True)
                response_parts: list[str] = []
                try:
                    result = Runner.run_streamed(agent, conversation_input, context=turn_context, max_turns=20)
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
    finally:
        driver.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
