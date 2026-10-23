#!/usr/bin/env python3
"""Run train_tracker.py for many clips on a few GPUs: K concurrent runs per GPU, the
next clip starts when a slot frees.

Each run's stdout goes to ``<clip>_<start time>_<tag>.log`` next to the status table
(default <data root>/queue/; the run dir does not exist yet when the trainer starts).
The status table (clip, gpu, pid, start, end, returncode, log) is rewritten at every
change; a crashed run is recorded and the queue moves on. SIGTERM or Ctrl-C to the queue
stops launching, forwards SIGTERM to the running trainers (they finalize their logs),
waits, exits.

    PY=.venv/bin/python  # from the repo root
    D=$MJLAB_DATA/queue
    nohup $PY tools/train_queue.py --gpus 0 1 --slots 2 --iters 30000 --tag full \\
        [--task Mjlab-Tracking-Flat-Unitree-G1-Vic-Freq] [--mps] \\
        [--status-file $D/<name>.tsv] > $D/<name>.log 2>&1 &

Clips default to every ``<clip>.npz`` in the motions dir, alphabetical. Arguments after
``--`` go to train_tracker.py.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from paths import MOTIONS_DIR, QUEUE_DIR, RUNS_DIR, VENV_BIN

HERE = Path(__file__).resolve().parent
STATUS_FILE = QUEUE_DIR / "queue_status.tsv"
COLUMNS = ("clip", "gpu", "pid", "start", "end", "returncode", "log")
POLL_SECONDS = 15


def now() -> str:
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(rows: list[dict[str, str]], path: Path) -> None:
  tmp_path = path.with_name(path.name + ".tmp")
  lines = ["\t".join(COLUMNS)]
  lines += ["\t".join(row.get(c, "") for c in COLUMNS) for row in rows]
  tmp_path.write_text("\n".join(lines) + "\n")
  os.replace(tmp_path, path)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--clips", nargs="+", help="default: every clip in --motions-dir, alphabetical"
  )
  parser.add_argument("--gpus", nargs="+", type=int, default=[0])
  parser.add_argument(
    "--slots", type=int, required=True, help="concurrent runs per GPU"
  )
  parser.add_argument("--iters", type=int, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--tag", required=True)
  parser.add_argument("--task", help="mjlab task id (default: train_tracker.py's)")
  parser.add_argument(
    "--mps", action="store_true", help="trainers are MPS clients (train_tracker --mps)"
  )
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--log-root", type=Path, default=RUNS_DIR)
  parser.add_argument(
    "--status-file",
    type=Path,
    default=STATUS_FILE,
    help="the stdout logs go next to it",
  )
  args, extra = parser.parse_known_args()
  if extra and extra[0] == "--":
    extra = extra[1:]

  clips = args.clips or sorted(p.stem for p in args.motions_dir.glob("*.npz"))
  rows = [{"clip": clip} for clip in clips]
  pending = list(range(len(rows)))
  # GPU-major interleave, so the first wave spreads evenly over the GPUs.
  free_slots = [gpu for _ in range(args.slots) for gpu in args.gpus]
  # Row index -> (process, gpu, log file).
  running: dict[int, tuple[subprocess.Popen, str, object]] = {}
  stopping = False

  def request_stop(signum, frame) -> None:
    nonlocal stopping
    stopping = True
    pids = [p.pid for p, _, _ in running.values()]
    print(f"[{now()}] signal {signum}: stop launching, SIGTERM to {pids}", flush=True)
    for process, _, _ in running.values():
      process.send_signal(signal.SIGTERM)

  signal.signal(signal.SIGTERM, request_stop)
  signal.signal(signal.SIGINT, request_stop)
  print(
    f"[{now()}] queue pid {os.getpid()}: {len(clips)} clips, "
    f"gpus {args.gpus} x {args.slots} slots",
    flush=True,
  )
  args.status_file.parent.mkdir(parents=True, exist_ok=True)
  write_status(rows, args.status_file)

  while (pending and not stopping) or running:
    while pending and free_slots and not stopping:
      index, gpu = pending.pop(0), free_slots.pop(0)
      clip = rows[index]["clip"]
      start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      log_path = args.status_file.parent / f"{clip}_{start}_{args.tag}.log"
      log_file = open(log_path, "w")
      command = [
        str(VENV_BIN / "python"), str(HERE / "train_tracker.py"),
        "--clip", clip, "--gpu", str(gpu),
        "--iters", str(args.iters), "--num-envs", str(args.num_envs),
        "--tag", args.tag,
        "--motions-dir", str(args.motions_dir), "--log-root", str(args.log_root),
        *(["--task", args.task] if args.task else []),
        *(["--mps"] if args.mps else []), *(["--", *extra] if extra else []),
      ]  # fmt: skip
      # Own session: a hangup or Ctrl-C of the queue's terminal does not reach the
      # trainers directly.
      process = subprocess.Popen(
        command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
      )
      running[index] = (process, str(gpu), log_file)
      rows[index].update(
        gpu=str(gpu), pid=str(process.pid), start=now(), log=str(log_path)
      )
      print(
        f"[{now()}] start {clip} gpu {gpu} pid {process.pid} -> {log_path}",
        flush=True,
      )
      write_status(rows, args.status_file)
      time.sleep(5)  # stagger env builds; distinct run dir timestamps
    time.sleep(POLL_SECONDS)
    for index in [i for i, (p, _, _) in running.items() if p.poll() is not None]:
      process, gpu, log_file = running.pop(index)
      log_file.close()
      free_slots.append(int(gpu))
      rows[index].update(end=now(), returncode=str(process.returncode))
      print(
        f"[{now()}] end {rows[index]['clip']} gpu {gpu} "
        f"returncode {process.returncode}",
        flush=True,
      )
      write_status(rows, args.status_file)
  print(f"[{now()}] queue done; {len(pending)} clips not started", flush=True)


if __name__ == "__main__":
  main()
