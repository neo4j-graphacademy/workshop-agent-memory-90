# Build a context graph

You build a small AI agent with a memory. The memory is a graph: entities,
the connections between them, and a record of what the agent did and why.
You do this on hosted infrastructure, so there is nothing to install beyond
a couple of command-line tools.

This is an individual exercise, not a team one. Everyone works through the
same 90-minute course, on their own machine, at their own pace.

This repo is the hands-on environment - the walkthrough itself is the
course your facilitator presents on the day. Open a Codespace here when
the course tells you to, and follow along there.

## Prerequisites

- A GitHub account, to open this repo in a Codespace (recommended—see
  below), or [uv](https://docs.astral.sh/uv/getting-started/installation/)
  and Node.js 18 or later if you'd rather run locally
- An OpenAI API key
- Your own Aura database credentials (given out by the facilitator on the
  day—nothing to provision yourself)
- About 90 minutes

## Run this in a Codespace

Everything in this workshop runs from a terminal. Rather than get everyone's
laptop into a working state, open this repo in a
[GitHub Codespace](https://github.com/codespaces)—Code -> Codespaces ->
Create codespace on main—and it gives you a browser-based terminal with
Node.js and [uv](https://docs.astral.sh/uv/getting-started/installation/)
already installed. Run every command below inside that terminal.

The commands are identical either way. The only difference is where they
run.

## What you'll build

No NAMS anywhere in this trajectory - self-hosted `neo4j-agent-memory`
throughout, over bolt in your generated app and over a local MCP server
for Copilot. Across the session, in order: connect this repo to your own
Aura database, generate a small full-stack agent app and seed it with
real data, wire Copilot to that same graph over MCP and watch it actually
remember things you tell it, then build a second, standalone agent of
your own - a small terminal coding assistant, backed by the same memory,
capable of reading, writing, and eventually updating its own code.

## Why a graph, and why hosted

An agent's memory is usually one of two things: a transcript, or a vector
index of embeddings. Neither tells you how two things are connected, or lets
you ask a question that spans several of them. A graph does both, and it
holds all three memory layers—the conversation, the entities and their
relationships, and the agent's own reasoning—as one connected structure,
not three separate stores.

Neo4j Aura is a fully-managed, cloud-hosted version of that graph. The
facilitator provisions your database ahead of time, so you get connection
credentials instead of something to install and run yourself—which is why
this exercise has no Docker step.

## What is in here

This repo is the hands-on environment, not the application itself. Early
in the course you'll generate the actual app in its own sibling folder,
outside this repo, using
[`create-context-graph`](https://github.com/neo4j-labs/create-context-graph).
Later, `reference/coding_assistant.py` is a working example of a small
standalone agent backed by the same memory package - a pattern to build
on, not something you're expected to read up front.

## For facilitators

- Send the OpenAI API key prerequisite out before the session—it can take a
  few minutes to provision.
- Provision one Aura database per attendee ahead of time, and download each
  one's credentials file from the Aura console. Hand these out at the start
  of the session—there is no shared database and nothing for attendees to
  provision themselves.
- Every attendee's app connects directly to Neo4j over Bolt, so ad-hoc
  Cypher and GDS both work—no caveats to give here.
