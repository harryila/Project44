#!/bin/bash
# Download and install the large files that are not kept in git (checkpoints, datasets, caches).
#
#   scripts/fetch_assets.sh quickstart          everything needed to transcribe and run the evaluations (about 1.6 GB)
#   scripts/fetch_assets.sh data                the training packs, PDMX and the gpu-branch corpora (about 4.5 GB)
#   scripts/fetch_assets.sh archive             candidate pools, outputs and every non-live checkpoint (about 22 GB)
#   scripts/fetch_assets.sh all
#   scripts/fetch_assets.sh <tier> --from-dir /path/with/downloaded/assets     (skip the download)
#   scripts/fetch_assets.sh <tier> --force      (overwrite files that already exist and differ)
#
# Every asset is verified against assets/SHA256SUMS before it is extracted. Archives extract into this
# project's root so the layout matches the one every script expects. Existing files are never overwritten
# unless --force is given; a differing existing file is reported and skipped.
#
# Requirements: gh (authenticated, with access to the release repository), zstd, tar, shasum.
# The release repository and tag are read from assets/RELEASE (one line: "<owner>/<repo> <tag>").
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${MUSICML_ROOT:-$(cd "$HERE/.." && pwd)}"
TIER="${1:-quickstart}"; shift || true
FROM_DIR=""; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from-dir) FROM_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
SUMS="$ROOT/assets/SHA256SUMS"
LIST="$ROOT/assets/ASSETS.csv"
[ -f "$SUMS" ] && [ -f "$LIST" ] || { echo "assets/SHA256SUMS or assets/ASSETS.csv missing" >&2; exit 1; }
read -r RELEASE_REPO RELEASE_TAG < "$ROOT/assets/RELEASE"
case "$RELEASE_REPO" in OWNER/REPO|"") echo "assets/RELEASE still has the placeholder; set it to the repository and tag that hosts the assets" >&2; [ -n "$FROM_DIR" ] || exit 1 ;; esac
DL="${FROM_DIR:-$ROOT/assets/_downloads}"
mkdir -p "$DL"
for tool in zstd tar shasum; do command -v "$tool" >/dev/null || { echo "missing tool: $tool (brew install $tool)" >&2; exit 1; }; done
if [ -z "$FROM_DIR" ]; then command -v gh >/dev/null || { echo "missing tool: gh (brew install gh; then gh auth login)" >&2; exit 1; }; fi

want_tier() {  # does asset tier $1 belong to requested tier $TIER
  case "$TIER" in
    all) return 0 ;;
    quickstart) [ "$1" = "quickstart" ] ;;
    data) [ "$1" = "data" ] ;;
    archive) [ "$1" = "archive" ] || [ "$1" = "archive-checkpoints" ] ;;
    *) echo "unknown tier $TIER (quickstart|data|archive|all)" >&2; exit 2 ;;
  esac
}

verify() {  # verify <file> <name>
  local expect; expect="$(grep "  $2\$" "$SUMS" | awk '{print $1}')"
  [ -n "$expect" ] || { echo "no checksum recorded for $2" >&2; return 1; }
  local got; got="$(shasum -a 256 "$1" | awk '{print $1}')"
  [ "$got" = "$expect" ] || { echo "CHECKSUM MISMATCH for $2" >&2; return 1; }
}

install_raw() {  # install_raw <downloaded file> <destination relative to ROOT>
  local dst="$ROOT/$2"
  if [ -f "$dst" ] && [ "$FORCE" != "1" ]; then
    if cmp -s "$1" "$dst"; then echo "  present   $2"; else echo "  DIFFERS   $2 (kept; use --force to replace)"; fi
    return 0
  fi
  mkdir -p "$(dirname "$dst")" && cp "$1" "$dst" && echo "  installed $2"
}

install_archive() {  # install_archive <file> <name>
  case "$2" in
    *.tar.zst) zstd -dc -q "$1" | tar -xf - -C "$ROOT" $( [ "$FORCE" = "1" ] || echo "-k" ) 2>>"$DL/extract.log" ;;
    *.zst)     local out="$ROOT/data/$(basename "$2" .zst)"; [ -f "$out" ] && [ "$FORCE" != "1" ] && { echo "  present   $out"; return 0; }; zstd -dc -q "$1" > "$out" ;;
  esac
  echo "  extracted $2"
}

echo "project root: $ROOT"
echo "release:      $RELEASE_REPO $RELEASE_TAG"
echo "tier:         $TIER"
# ASSETS.csv columns: asset,tier,size_bytes,sha256,extracts_to,contents (quoted CSV; the first five fields have no commas)
tail -n +2 "$LIST" | while IFS=, read -r NAME ATIER SIZE SHA DEST REST; do
  want_tier "$ATIER" || continue
  base="$(basename "$NAME")"
  f="$DL/$base"
  if [ ! -f "$f" ]; then
    [ -n "$FROM_DIR" ] && { echo "  MISSING in --from-dir: $base" >&2; continue; }
    echo "  download  $base ($((SIZE/1000000)) MB)"
    gh release download "$RELEASE_TAG" -R "$RELEASE_REPO" -p "$base" -D "$DL" --clobber || { echo "  download failed: $base" >&2; continue; }
  fi
  verify "$f" "$NAME" || continue
  case "$NAME" in
    ckpt-archive/*) install_raw "$f" "$DEST" ;;
    *.ckpt)         install_raw "$f" "$DEST" ;;
    PDMX_mxl.tar.gz|PDMX.csv.zst)
      PD="${MUSICML_PDMX_ROOT:-${MUSICML_DATASETS:-$HOME/datasets}/pdmx}"; mkdir -p "$PD"
      if [ "$NAME" = "PDMX.csv.zst" ]; then [ -f "$PD/PDMX.csv" ] && [ "$FORCE" != "1" ] && echo "  present   $PD/PDMX.csv" || { zstd -dc -q "$f" > "$PD/PDMX.csv"; echo "  extracted $PD/PDMX.csv"; }
      else [ -d "$PD/mxl" ] && [ "$FORCE" != "1" ] && echo "  present   $PD/mxl" || { tar -xzf "$f" -C "$PD"; echo "  extracted $PD/mxl"; }; fi ;;
    *)              install_archive "$f" "$NAME" ;;
  esac
done
echo "done. Checkpoints land in MIDI2ScoreTransformer/checkpoints/, data in data/ and legacy/data/, PDMX under \${MUSICML_DATASETS:-~/datasets}/pdmx."
echo "ASAP: the quickstart tier installs the pinned TimFelixBeyer/asap-dataset checkout (commit 8cba199e) together with the 967 generated chunk files and the tokenization cache; no separate clone is needed."
