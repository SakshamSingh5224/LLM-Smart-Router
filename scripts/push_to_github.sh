#!/usr/bin/env bash
# Commit and push this project to GitHub.
#
#   bash scripts/push_to_github.sh https://github.com/<user>/<repo>.git   # existing empty repo
#   bash scripts/push_to_github.sh                                        # uses GitHub CLI (gh) to create the repo
#
# Optional env: REPO_NAME (default llm-smart-router), VISIBILITY (public|private, default public)
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_URL="${1:-}"
REPO_NAME="${REPO_NAME:-llm-smart-router}"
VISIBILITY="${VISIBILITY:-public}"

git config user.name  >/dev/null || { echo "Set your identity first:  git config --global user.name 'Your Name'"; exit 1; }
git config user.email >/dev/null || { echo "Set your identity first:  git config --global user.email 'you@example.com'"; exit 1; }

[ -d .git ] || git init
git branch -M main
git add -A

# never publish secrets
if git ls-files --error-unmatch .env >/dev/null 2>&1 || git diff --cached --name-only | grep -qx '.env'; then
  echo "ERROR: .env is staged. Run: git rm --cached .env"; exit 1
fi

git diff --cached --quiet || git commit -m "Phase 1: local SLM router, free-tier model setup, RouteLLM dataset prep"

if [ -n "$REMOTE_URL" ]; then
  git remote get-url origin >/dev/null 2>&1 && git remote set-url origin "$REMOTE_URL" || git remote add origin "$REMOTE_URL"
  git push -u origin main
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin --push
else
  cat <<MSG
No remote given and GitHub CLI (gh) is not logged in. Do ONE of:

  A) Create an EMPTY repo at https://github.com/new (no README), then run:
       bash scripts/push_to_github.sh https://github.com/<your-user>/<repo>.git
     When git asks for a password, paste a Personal Access Token
     (GitHub -> Settings -> Developer settings -> Tokens (classic) -> scope: repo).

  B) Install + log in to gh, then re-run this script with no arguments:
       sudo apt install -y gh && gh auth login
MSG
  exit 1
fi
echo "Pushed."
