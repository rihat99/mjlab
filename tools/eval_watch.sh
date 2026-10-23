#!/usr/bin/env bash
# Evaluate every new model_<it>.pt of a SuperMimic run on its library as it appears
# (train_library.py --eval), until the training process has exited and every checkpoint
# has its eval_library_model_<it>.json.
# Usage: [GPU=0] [PY=<python>] [TASK=<task id>] [EVAL_ARGS="<flags>"] \
#   tools/eval_watch.sh <run_dir> <train_pid>
# PY defaults to the worktree's .venv/bin/python, TASK to train_library.py's default.
# EVAL_ARGS go to evaluate_library, e.g. "--motion-files <subset>.txt" to evaluate a
# large library's checkpoints on a subset while training.
set -u
RUN=$1; PID=$2
TOOLS=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$TOOLS/../.venv/bin/python}
while true; do
  for ckpt in $(ls "$RUN"/model_*.pt 2>/dev/null | sort -V); do
    it=$(basename "$ckpt" .pt); it=${it#model_}
    [ -f "$RUN/eval_library_model_$it.json" ] && continue
    sleep 5  # let the trainer finish writing
    echo "$(date +%H:%M:%S) eval $it"
    "$PY" "$TOOLS/train_library.py" --eval --gpu "${GPU:-0}" ${TASK:+--task "$TASK"} --run-dir "$RUN" --checkpoint "$ckpt" \
      ${EVAL_ARGS:+-- $EVAL_ARGS} 2>&1 \
      | grep -E "normalised return per clip length|Error|error" | tail -2
  done
  if ! kill -0 "$PID" 2>/dev/null; then
    pending=$(for c in "$RUN"/model_*.pt; do i=$(basename "$c" .pt); i=${i#model_}; [ -f "$RUN/eval_library_model_$i.json" ] || echo x; done | wc -l)
    [ "$pending" -eq 0 ] && { echo "$(date +%H:%M:%S) done"; break; }
  fi
  sleep 60
done
