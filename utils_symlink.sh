#!/usr/bin/env bash
# =============================================================================
# link_uaspeech.sh
# Creates symlinks for every .wav in the UASpeech audio folder into one flat dir.
#
# Directory structure handled:
#   audio/
#     control/
#       CF01/   *.wav          ← 2-level deep (control sub-folders)
#       CM01/   *.wav
#       ...
#     F02/      *.wav          ← 1-level deep
#     F03/      *.wav
#     M04/      *.wav
#     ...
#
# Usage:
#   ./link_uaspeech.sh [SOURCE_AUDIO_DIR] [TARGET_DIR]
#
#   SOURCE_AUDIO_DIR  – path to the UASpeech "audio" folder  (default: ./audio)
#   TARGET_DIR        – flat output directory for symlinks    (default: ./wav_all)
# =============================================================================

set -euo pipefail

# ── Arguments / defaults ─────────────────────────────────────────────────────
SOURCE_DIR="${1:-./audio}"
TARGET_DIR="${2:-./wavs}"

# ── Validate source ───────────────────────────────────────────────────────────
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "[ERROR] Source directory not found: $SOURCE_DIR" >&2
    exit 1
fi

# Resolve to absolute path so symlinks are not relative-path-broken
SOURCE_ABS="$(realpath "$SOURCE_DIR")"
mkdir -p "$TARGET_DIR"
TARGET_ABS="$(realpath "$TARGET_DIR")"

echo "============================================="
echo "  UASpeech WAV Symlinker"
echo "============================================="
echo "  Source : $SOURCE_ABS"
echo "  Target : $TARGET_ABS"
echo "---------------------------------------------"

# ── Counters ──────────────────────────────────────────────────────────────────
linked=0
skipped=0
conflict=0

# ── Main loop ─────────────────────────────────────────────────────────────────
# -print0 / read -d '' handles filenames with spaces or special characters
while IFS= read -r -d '' wav; do
    filename="$(basename "$wav")"
    link="$TARGET_ABS/$filename"

    if [[ -L "$link" ]]; then
        # Symlink already exists – skip silently
        (( skipped++ )) || true

    elif [[ -e "$link" ]]; then
        # A real file (not a symlink) has the same name – warn and skip
        echo "[CONFLICT] Real file already at target, skipping: $filename"
        (( conflict++ )) || true

    else
        ln -s "$wav" "$link"
        (( linked++ )) || true
    fi

done < <(find "$SOURCE_ABS" -type f \( -iname "*.wav" \) -print0 | sort -z)

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================="
echo "  Done."
echo "  Symlinks created : $linked"
echo "  Already existed  : $skipped"
echo "  Conflicts (skipped): $conflict"
echo "============================================="