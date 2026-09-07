#!/usr/bin/env bash
# Runs as onCreateCommand, captured into the Codespaces prebuild snapshot.
# Warms every cache create-context-graph's self-hosted scaffold needs, so
# `make install` and first use hit cache instead of downloading a multi-GB
# ML stack (and 615 npm packages) live during the workshop.
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# uv installs from cache via hardlink when the cache and the target venv
# share a filesystem -- near-instant. /workspaces and $HOME are different
# mounts in this devcontainer image, so a cache under $HOME (uv's default)
# silently falls back to a full byte copy per package once the generated
# app's own .venv lives under /workspaces, which is where the slowdown was
# actually coming from. devcontainer.json's containerEnv sets this for
# every later shell too; exported here so this script is correct standalone.
export UV_CACHE_DIR=/workspaces/.uv-cache

# Full dependency set backend/pyproject.toml.j2 writes for --self-hosted --
# torch, spaCy, GLiNER, sentence-transformers included (spaCy's package
# still installs even though its model download step is gone -- pyproject
# declares it as part of the `extraction` extra). Warms uv's wheel cache,
# which the real `uv sync` inside the generated app reuses. This is a
# GitHub Actions-backed Codespaces prebuild -- everything below runs once,
# off the critical path, before any learner opens the codespace, so warming
# it costs nothing even for stages the workshop ends up not using at
# runtime (GLiNER -- see configure-app.py). More in cache is free; don't
# remove a warm step just because the matching runtime stage got disabled.
uv venv /tmp/warm --python 3.12
# Activate rather than call /tmp/warm/bin/python directly -- needed for
# VIRTUAL_ENV to be set so `uv pip install` (no --python flag below)
# targets this venv instead of erroring with no target found.
source /tmp/warm/bin/activate

uv pip install "neo4j-agent-memory[litellm,sentence-transformers,extraction,fuzzy,openai]"

# task 10's reference/coding_assistant.py runs via `uv run --script` (PEP
# 723, inline deps, no venv of its own) so it needs the exact same package
# set warmed here too -- UV_CACHE_DIR above is shared across every uv
# invocation, this venv included, so wheels warmed here are reused by that
# script's own ephemeral environment at workshop time.
uv pip install "neo4j-agent-memory[openai]==0.5.0" openai-agents python-dotenv

# No spaCy model warm here -- `install-backend` no longer downloads one
# (see configure-app.py). Pinning it to `--direct` fixed the cache-hit
# (confirmed: no re-download) but not the wait, which turned out to be
# `uv run`'s per-invocation environment-sync check plus spaCy's own
# cold-import cost (thinc/blis/cython), paid twice (once for the import
# check, once for the download command) -- neither is network-bound, so no
# amount of warming fixes it. spaCy is unused at runtime; dropping the step
# is the only way to remove the wait rather than just make it reliable.

# sentence-transformers and GLiNER pull their weights from the Hugging Face
# Hub on first *use*, not at pip-install time -- the next latency spike if
# left alone. Pre-download the exact models create-context-graph's
# self-hosted default loads (memory.py.j2, gliner_extractor.py). Warmed
# regardless of the runtime GLiNER/spaCy disable above -- free at prebuild
# time, and keeps this script correct if that runtime toggle ever changes.
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"
python -c "
from gliner import GLiNER
GLiNER.from_pretrained('gliner-community/gliner_medium-v2.5')
"

deactivate
rm -rf /tmp/warm

# Frontend deps -- create-context-graph's frontend/package.json.j2 has no
# per-domain/per-framework variation, so this exact dependency set is what
# every learner's `npm install` resolves regardless of which domain they
# pick. Keep this list in sync with create-context-graph's
# frontend/package.json.j2 if it changes.
#
# The generator ships no lockfile, so a plain `npm install` at workshop time
# doesn't just fetch cached tarballs -- it re-resolves the whole dependency
# graph against the registry (metadata for ~615 packages), runs a live
# audit call, and pings the registry for npm's own update-notifier. All of
# that is uncacheable network work bolted onto an otherwise cache-hit
# install -- the same shape as spaCy's compatibility-check bug above. Fix:
# resolve once here, keep the resulting lockfile (captured in the prebuild
# snapshot same as the caches, since it's under /workspaces), and have
# configure-app.py drop it into the generated app so `install-frontend` can
# run `npm ci` -- a frozen install from a known-good lock, no resolution,
# no update-notifier ping observed once warm.
export NPM_CONFIG_CACHE=/workspaces/.npm-cache
mkdir -p /tmp/warm-frontend
cat > /tmp/warm-frontend/package.json << 'EOF'
{
  "name": "warm-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@chakra-ui/react": "^3.32.0",
    "@emotion/react": "^11.14.0",
    "lucide-react": "^0.563.0",
    "@neo4j-nvl/react": "^1.1.0",
    "@neo4j-nvl/base": "^1.1.0",
    "neo4j-driver": "^6.0.1",
    "next": "^15.3.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.52.0",
    "@types/node": "^22.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "eslint": "^9.0.0",
    "eslint-config-next": "^15.3.0",
    "typescript": "^5.7.0"
  },
  "overrides": {
    "lodash": "^4.17.24",
    "postcss": "^8.5.10"
  }
}
EOF
(cd /tmp/warm-frontend && npm install --no-audit --no-fund --loglevel=error)

# Persist the resolved lockfile under /workspaces so it's captured in the
# same prebuild snapshot as the caches -- configure-app.py reads it from
# here at workshop time.
cp /tmp/warm-frontend/package-lock.json /workspaces/.frontend-package-lock.json
rm -rf /tmp/warm-frontend
