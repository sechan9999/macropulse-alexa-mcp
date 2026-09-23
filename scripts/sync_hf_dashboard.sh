#!/usr/bin/env bash
# Mirror this repo's code into a checkout of sechan9999/hf-macro-dashboard (the repo Streamlit Cloud
# deploys hf-macro-dashboard.streamlit.app from). One-way: macropulse-alexa-mcp is the source of truth.
#
#   scripts/sync_hf_dashboard.sh <source checkout> <destination checkout>
#
# Kept out of the mirror (never copied, never deleted in the destination):
#   signals/                              each repo's daily quant-signal bot writes its own snapshots
#   .github/workflows/sync-hf-dashboard.yml   this sync job only runs in the source repo
# Anything else that exists only in the destination is deleted, so edit code here, not there.
# Stages the result in the destination; the caller decides whether to commit.
set -euo pipefail

src="${1:?source checkout}"
dst="${2:?destination checkout}"

rsync -a --delete \
  --exclude '.git/' \
  --exclude 'signals/' \
  --exclude '.github/workflows/sync-hf-dashboard.yml' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "${src%/}/" "${dst%/}/"

git -C "$dst" add -A
