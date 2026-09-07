# Orientation

This workspace gives you two completely separate sources of information.
Before answering anything, work out which one a question is actually
about, and say so at the start of your answer, not buried in an
explanation afterward.

## 1. The repo

This is the actual codebase: the files under this workspace root -
`scripts/`, `README.md`, `.devcontainer/`, `.vscode/`, and anything you
or the learner create later in the session (for example, a custom agent
script built in a later lesson, or edits to that script in a bonus
step). Real, current, ground truth. Read and edit it with your normal
file tools.

This repo is not a production system. It has no services, deployments, or
incidents of its own. If a question is about *this codebase* - what a
script does, what a lesson asks for, what to change - answer from the
files, not from the memory graph.

## 2. The graph

A Neo4j Aura database behind the `neo4j-agent-memory` package, reachable
only through the `memory` MCP server's tools below - never through file
reads, and never part of this repo's own files. It holds two different
kinds of thing, both real graph data but not the same kind of fact:

- **Prior context data.** An earlier task ran `create-context-graph` to
  populate this graph with entities, documents, and decision
  traces from an already extant domain context graph -- a domain the learner picked (services, repositories,
  incidents, deployments - names like "user-api" or "cdn-proxy" are
  from earlier work in the graph. Nothing retrieved from the graph
  about these entities describes anything in this repo, not this
  Codespace. It has been prepopulated to demonstrate what a complete context graph looks like.
- **Real session memory.** Every message sent to the learner's own
  generated app (module 1) got written into this same graph - short-term
  conversation history, entities/facts/preferences extracted from what was
  said, and reasoning traces. Two independent readers and writers, one
  graph.

## How to interact with each

- When the user asks you questions, first consider whether they are asking about the current repo, or the repo represented in the graph. Then call the appropriate tools to provide a legitimate answer.
- On every interaction, ensure you use your tools. For example, you should ALWAYS use get_context() to retrieve and you should ALWAYS record every message and entity.

## Tool map (`memory` MCP server, 16 tools, extended profile)

Core:

| Tool | Purpose |
| --- | --- |
| `memory_search` | Hybrid vector + graph search across messages, entities, preferences, traces |
| `memory_get_context` | Assembled context (short-term + long-term + reasoning) for a session - call this first for "what do you know about X" |
| `memory_store_message` | Store a message in conversation memory |
| `memory_add_entity` | Create or update an entity in the knowledge graph |
| `memory_add_preference` | Record a user preference |
| `memory_add_fact` | Store a subject-predicate-object fact |
| `memory_start_trace` / `memory_record_step` / `memory_complete_trace` | Record a reasoning trace for a multi-step task | You should call this every time you use any tool, to maintain a complete reasoning history

Extended:

| Tool | Purpose |
| --- | --- |
| `memory_get_conversation` | Full conversation history for a session |
| `memory_list_sessions` | List available sessions with previews |
| `memory_get_entity` | Detailed entity info plus its graph relationships |
| `memory_export_graph` | Export a subgraph as JSON |
| `memory_create_relationship` | Create a typed relationship between two entities |
| `memory_get_observations` | Observations and extracted insights for a session |
| `graph_query` | Read-only Cypher against the graph - writes are rejected |

`graph_query` is the only raw-Cypher tool; use it when the structured tools
above don't cover the question, not as a first resort.
