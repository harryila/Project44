#!/bin/bash
# Source this from any launcher: `. "$(dirname "$0")/_env.sh"`
# Sets REPO (project root), PY (the python of venv311 or the active interpreter), and the GPU knobs.
# Works with bash 3.2 (macOS) and bash 4+.
_here="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
REPO="${MUSICML_ROOT:-$(cd "$_here/.." && pwd)}"
if [ -x "$REPO/venv311/bin/python" ]; then PY="$REPO/venv311/bin/python"; else PY="$(command -v python3)"; fi
MUSICML_GPU="${MUSICML_GPU:-0}"
KILL_EXISTING="${KILL_EXISTING:-0}"
LOG_DIR="${MUSICML_LOG_DIR:-$REPO/logs}"
mkdir -p "$LOG_DIR"
export REPO PY MUSICML_GPU KILL_EXISTING LOG_DIR
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer${PYTHONPATH:+:$PYTHONPATH}"

# kill_existing_train: only when the caller opted in. On a shared machine `pkill -f train.py`
# would kill other people's runs, so the default is to refuse to start if one is running.
kill_existing_train() {
  if pgrep -f '[t]rain.py fit' > /dev/null 2>&1; then
    if [ "$KILL_EXISTING" = "1" ]; then
      pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
    else
      echo "A train.py run is already active. Set KILL_EXISTING=1 to replace it, or wait." >&2
      return 1
    fi
  fi
  return 0
}
