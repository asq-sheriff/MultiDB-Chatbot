#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"

# If nothing changed (staged or unstaged), skip
if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to commit. Skipping."
  exit 0
fi

# Ensure we’re on the intended branch
git checkout "$BRANCH"

# Stage and commit
git add -A
git commit -m "chore(codex): automated changes [skip ci]"

# Rebase in case of parallel pushes
git pull --rebase origin "$BRANCH" || true

# Push
git push origin "$BRANCH"

# Optional: auto-tag this run if CODEX_TAG is set in the env
if [[ -n "${CODEX_TAG:-}" ]]; then
  git tag -a "$CODEX_TAG" -m "codex tag: $CODEX_TAG"
  git push origin "$CODEX_TAG"
fi

echo "✅ Codex publish complete."
