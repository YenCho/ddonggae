#!/usr/bin/env bash
# Download the trained perception models into perception/models/.
#
# The weights are NOT in this git repository. They are published as GitHub
# Release assets so that a plain `git clone` stays small and nobody burns Git LFS
# bandwidth. They are also licensed differently from this repo: the models are
# Ultralytics YOLO derivatives and are therefore AGPL-3.0, not MIT. See NOTICE.
#
# Usage:
#   scripts/fetch_models.sh                # fetch the pinned release
#   RELEASE=v1.0.0 scripts/fetch_models.sh # fetch a specific release
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/perception/models"

GH_REPO="${GH_REPO:-YenCho/ddonggae}"
RELEASE="${RELEASE:-v1.0.0}"
BASE="https://github.com/${GH_REPO}/releases/download/${RELEASE}"

# asset name  ->  path inside perception/models/
# The first three are the weights the robot actually ran in the two finals
# (face fine-tune of 2026-07-24, BP verifier retrained on real photographs; the
# AO verifier is loaded but its route is disabled by --pair bp). The rest are
# the earlier generations, kept so the A/B comparisons in the docs can be
# reproduced with FACE_WEIGHTS= / AO_WEIGHTS= / BP_WEIGHTS= overrides.
declare -A ASSETS=(
  ["a1_objectseg_best.pt"]="a1_objectseg/best.pt"
  ["unified_face_ft_20260724_best.pt"]="unified_face_ft_20260724/best.pt"
  ["verifiers_pair_banana_pineapple_real_v1.onnx"]="verifiers/pair_banana_pineapple_real_v1.onnx"
  ["verifiers_pair_apple_orange_real_20260724.onnx"]="verifiers/pair_apple_orange_real_20260724.onnx"
  ["unified_face_best.pt"]="unified_face/best.pt"
  ["verifiers_pair_apple_orange_v2_anchor.onnx"]="verifiers/pair_apple_orange_v2_anchor.onnx"
  ["verifiers_pair_apple_orange_v2.onnx"]="verifiers/pair_apple_orange_v2.onnx"
  ["verifiers_pair_banana_pineapple_v2.onnx"]="verifiers/pair_banana_pineapple_v2.onnx"
)

echo "Fetching models from ${GH_REPO} @ ${RELEASE}"
for asset in "${!ASSETS[@]}"; do
  target="${DEST}/${ASSETS[$asset]}"
  mkdir -p "$(dirname "${target}")"
  if [[ -f "${target}" ]]; then
    echo "  = ${ASSETS[$asset]} (already present, skipping)"
    continue
  fi
  echo "  + ${ASSETS[$asset]}"
  if ! curl -fL --progress-bar "${BASE}/${asset}" -o "${target}"; then
    rm -f "${target}"
    echo "" >&2
    echo "Could not download ${asset} from ${BASE}." >&2
    echo "If this repository is still private, release assets need auth - use:" >&2
    echo "    gh release download ${RELEASE} --repo ${GH_REPO} --dir ${DEST}" >&2
    echo "and then re-run this script to verify the checksums." >&2
    exit 1
  fi
done

# Verify integrity. SHA256SUMS is published alongside the weights.
if curl -fsL "${BASE}/SHA256SUMS" -o "${DEST}/.SHA256SUMS" 2>/dev/null; then
  echo "Verifying checksums..."
  fail=0
  while read -r want name; do
    target="${DEST}/${ASSETS[$name]:-}"
    [[ -n "${ASSETS[$name]:-}" && -f "${target}" ]] || continue
    got="$(sha256sum "${target}" | cut -d' ' -f1)"
    if [[ "${got}" == "${want}" ]]; then
      echo "  ok  ${ASSETS[$name]}"
    else
      echo "  BAD ${ASSETS[$name]}"
      echo "      expected ${want}"
      echo "      got      ${got}"
      fail=1
    fi
  done < "${DEST}/.SHA256SUMS"
  rm -f "${DEST}/.SHA256SUMS"
  [[ "${fail}" -eq 0 ]] || { echo "Checksum verification FAILED."; exit 1; }
fi

echo "Done. Models are in ${DEST}"
