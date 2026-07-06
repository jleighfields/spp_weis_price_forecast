#!/usr/bin/env bash
# Regenerate requirements.txt for the Posit Connect Cloud deploy from
# pyproject.toml.
#
# Why this exists (the non-obvious bits, so nobody re-derives them):
#   - Connect Cloud installs Python deps from requirements.txt in the GitHub
#     repo; it does NOT read pyproject.toml or uv.lock. So requirements.txt is
#     the source of truth for the deploy and must track pyproject.toml.
#   - Connect Cloud runs on Linux x86_64. This dev box is aarch64, and
#     pyproject pins torch to a local cu128 index (GB10 GPU). A plain
#     `uv export` / `pip freeze` here would bake in wrong (aarch64 / +cu128)
#     pins, so we resolve explicitly for the deploy target instead:
#       --python-platform x86_64-unknown-linux-gnu   (Connect Cloud's platform)
#       --no-sources                                 (ignore the cu128 index;
#                                                     use plain PyPI wheels)
#   - torch is pinned to the version that TRAINED the champion. A Darts/torch
#     checkpoint saved by torch X loads cleanly under torch X; drifting the
#     serving torch risks a load failure. Defaults to the local torch version;
#     override with TORCH_VERSION=x.y.z.
#
# manifest.json is intentionally NOT touched — Connect Cloud does not use it
# for Python content (it is R-only there).
#
# Usage:
#   scripts/gen_deploy_requirements.sh
#   TORCH_VERSION=2.11.0 scripts/gen_deploy_requirements.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_VERSION="${TORCH_VERSION:-$(uv run python -c 'import torch; print(torch.__version__.split("+")[0])')}"
echo "resolving deploy requirements for x86_64 linux, torch==${TORCH_VERSION}"

uv pip compile pyproject.toml \
  --quiet \
  --no-sources \
  --python-platform x86_64-unknown-linux-gnu \
  --python-version 3.11 \
  --override <(echo "torch==${TORCH_VERSION}") \
  -o requirements.txt

echo "wrote requirements.txt ($(grep -c '==' requirements.txt) pinned packages)"
grep -iE '^(darts|torch|torchmetrics|lightning|pytorch-lightning|shiny)==' requirements.txt
