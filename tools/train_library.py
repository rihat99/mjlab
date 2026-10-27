"""Train, play or evaluate one SuperMimic tracker on a motion library, fully local.

The task is ``Mjlab-Tracking-Flat-Unitree-G1-SuperMimic`` (``TASK``).
Train on the library = the kept clips of <motions-dir>/manifest.json if there is one
(tools/filter_seed.py), else every npz of <motions-dir> (sorted), or the ``--clips``
subset; <motions-dir> defaults to BONES-SEED (data/motions/seed), <log-root> to
runs/seed (``tools/paths.py``). Run dir <log-root>/<timestamp>[_<tag>]/ holding
params/, TensorBoard events, metrics.csv, curves.png and model_<it>.pt (every 4000
iterations and at the end).
The motion paths go to mjlab as a list file, <log-root>/library_<sha256 of the sorted
path list>[:12]_files.txt (one path per line; a tuple literal of 129k paths would
exceed the argument size limit), which params/env.yaml records. The library itself is
recorded once in <log-root>/library_<same hash>.txt, in ``sha256sum`` format
(``sha256sum -c`` re-checks the files); training refuses to start if that record
exists but the files have changed since:

    PY=<worktree>/.venv/bin/python
    $PY tools/train_library.py --iters 160000 --tag seed
    $PY tools/train_library.py --iters 14000 --motions-dir data/motions/lafan \\
        --log-root runs/lafan [--clips walk1_subject1,...]

Evaluate: the paper's library protocol (each clip once from frame 0, in waves of at
most --num-envs clips) on the run's training library ->
<run dir>/eval_library_model_<it>.json, and a
one-line summary (normalised return per clip length and per survived step, each as
mean over clips and frame-weighted; failed clips), for --checkpoint, else for every
checkpoint of the run dir (default: the newest run) that has no JSON yet
(``-- --overwrite True``: every one), in one process. ``--watch`` also evaluates
each checkpoint the training writes, until the training process --pid has exited
(it first waits for the run's params/env.yaml, written once the training env is
built).
Play headless (no viewer): one clip of the run's library, records
<run dir>/videos/play/rl-video-step-0.mp4 (overwritten by the next play of that run),
one clip pass by default; it defaults to the newest run and its last checkpoint:

    $PY tools/train_library.py --eval [--run-dir DIR | --checkpoint FILE]
    $PY tools/train_library.py --eval --watch --run-dir DIR --pid TRAIN_PID
    $PY tools/train_library.py --play --clip walk1_subject1 [--run-dir DIR] \\
        [--checkpoint FILE] [--video-length N]

Play and evaluate use the library the run's params/env.yaml records (its clip list or
tuple of npz paths); if that list or one of its files is not on this machine (a run
made elsewhere, e.g. the shipped checkpoints/seed), they stop and ask for
``--motions-dir``, whose library (as for training) then replaces the recorded one.

Anything after ``--`` is passed to mjlab's train / play / evaluate_library unchanged
(e.g. ``-- --agent.save-interval 100``; a repeated flag overrides the launcher's).
``--print-cmd`` prints the command and exits. The mjlab process replaces this one
(exec), so a nohup'ed PID is the trainer's PID. The mjlab commands are the ones
installed next to the interpreter that runs this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from paths import LOG_ROOT, MOTIONS_DIR

from mjlab.tasks.tracking.mdp.commands import expand_motion_files

TASK = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"


class ParamsLoader(yaml.SafeLoader):
  """Reads mjlab's params/env.yaml as plain data.

  Tuples become lists, other python/* tagged nodes None.
  """


def construct_tuple(loader: ParamsLoader, node: yaml.SequenceNode) -> list[Any]:
  """A python/tuple node as a list."""
  return loader.construct_sequence(node)


ParamsLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_tuple)
ParamsLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def library_record(motions: list[Path], log_root: Path) -> Path:
  """``<log_root>/library_<sha256 of the sorted path list>[:12].txt``."""
  listing = "".join(f"{path}\n" for path in motions)
  return log_root / f"library_{hashlib.sha256(listing.encode()).hexdigest()[:12]}.txt"


def write_library_record(motions: list[Path], record: Path) -> None:
  """Write ``record`` (``sha256sum`` lines of ``motions``) unless it already exists.

  Raises if the record exists with other file hashes: the motions changed since a run
  trained on that list.
  """
  content = "".join(
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n" for path in motions
  )
  if record.exists():
    if record.read_text() != content:
      raise SystemExit(
        f"{record} lists other file hashes: the motions changed since it was "
        f"written (check with sha256sum -c {record})"
      )
    return
  record.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = record.with_name(f"{record.name}.{os.getpid()}.tmp")
  tmp_path.write_text(content)
  tmp_path.chmod(0o444)
  os.replace(tmp_path, record)


def library_list(motions: list[Path], log_root: Path) -> Path:
  """``<log_root>/library_<hash>_files.txt``, the hash of ``library_record``."""
  record = library_record(motions, log_root)
  return record.with_name(f"{record.stem}_files.txt")


def write_library_list(motions: list[Path], path: Path) -> None:
  """Write ``motions`` to ``path``, one per line, unless it exists (mjlab's
  ``motion_file`` takes the list in place of a tuple of paths)."""
  if path.exists():
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
  tmp_path.write_text("".join(f"{motion}\n" for motion in motions))
  tmp_path.chmod(0o444)
  os.replace(tmp_path, path)


def kept_clips(motions_dir: Path) -> list[Path]:
  """The kept clips of ``<motions_dir>/manifest.json``, else every npz (sorted)."""
  manifest = motions_dir / "manifest.json"
  if not manifest.is_file():
    return sorted(motions_dir.glob("*.npz"))
  clips = json.loads(manifest.read_text())["clips"]
  return sorted(
    motions_dir / f"{name}.npz" for name, clip in clips.items() if clip["kept"]
  )


def latest_checkpoint(run_dir: Path) -> Path:
  """``model_<it>.pt`` with the highest ``it`` in ``run_dir``."""
  checkpoints = [
    p for p in run_dir.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)
  ]
  if not checkpoints:
    raise FileNotFoundError(f"no model_<it>.pt in {run_dir}")
  return max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))


def resolve_run_dir(args: argparse.Namespace) -> Path:
  """``args.run_dir``, else the newest run under ``args.log_root``."""
  if args.run_dir is not None:
    return args.run_dir.resolve()
  runs = sorted(p for p in args.log_root.iterdir() if p.is_dir())  # timestamped
  if not runs:
    raise FileNotFoundError(f"no runs under {args.log_root}")
  return runs[-1]


def resolve_checkpoint(args: argparse.Namespace) -> Path:
  """``args.checkpoint``, else the last checkpoint of ``resolve_run_dir(args)``."""
  if args.checkpoint is not None:
    return args.checkpoint.resolve()
  return latest_checkpoint(resolve_run_dir(args))


def wait_for_params(run_dir: Path, pid: int) -> None:
  """Wait until mjlab has written ``run_dir/params/env.yaml`` (once the training env
  is built: seconds on LAFAN1, 7-10 min on BONES-SEED) while process ``pid`` runs."""
  params = run_dir / "params" / "env.yaml"
  while not params.is_file():
    if not Path(f"/proc/{pid}").exists():
      raise SystemExit(f"[train_library] process {pid} ended before writing {params}")
    time.sleep(10)


def recorded_library(run_dir: Path) -> tuple[list[Path], Path | None]:
  """The motion files a run trained on, from its ``params/env.yaml``, and the ``.txt``
  clip list that records them (None for a recorded tuple of paths).

  Exits with a message asking for ``--motions-dir`` when the record, its clip list or
  one of its files is not on this machine.
  """
  params = run_dir / "params" / "env.yaml"
  if not params.is_file():
    raise SystemExit(f"{params} not found: choose the library with --motions-dir")
  env_cfg = yaml.load(params.read_text(), Loader=ParamsLoader)
  recorded = env_cfg["commands"]["motion"]["motion_file"]
  clip_list = None
  if isinstance(recorded, str) and recorded.endswith(".txt"):
    clip_list = Path(recorded)
    if not clip_list.is_file():
      raise SystemExit(
        f"the clip list {clip_list} of {run_dir} is not on this machine: choose the "
        "library with --motions-dir"
      )
  library = [Path(path) for path in expand_motion_files(recorded)]
  missing = [str(path) for path in library if not path.is_file()]
  if missing:
    raise SystemExit(
      f"{len(missing)} of {len(library)} library files of {run_dir} not found (e.g. "
      f"{missing[0]}): choose the library with --motions-dir"
    )
  return library, clip_list


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
    "--play", action="store_true", help="headless video of a checkpoint on one --clip"
  )
  mode.add_argument(
    "--eval",
    action="store_true",
    help="library metrics JSON of --checkpoint, else of every checkpoint of the run",
  )
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--iters", type=int, help="training iterations (training)")
  parser.add_argument(
    "--num-envs",
    type=int,
    default=4096,
    help="training envs (eval: clips played at once, play: 1)",
  )
  parser.add_argument("--tag", default="", help="suffix of the run dir name")
  parser.add_argument(
    "--clips", help="train: comma-separated clip ids of <motions-dir> (default: all)"
  )
  parser.add_argument("--clip", help="play: clip id of the run's library")
  parser.add_argument(
    "--motions-dir",
    type=Path,
    help=f"npz folder of the library (train default: {MOTIONS_DIR}; play/eval: "
    "replaces the run's recorded library)",
  )
  parser.add_argument(
    "--log-root",
    type=Path,
    default=LOG_ROOT,
    help="folder of the runs (default: %(default)s; LAFAN1: runs/lafan)",
  )
  parser.add_argument(
    "--run-dir", type=Path, help="play/eval: run dir (default: newest run)"
  )
  parser.add_argument(
    "--checkpoint", type=Path, help="play/eval: checkpoint (default: last in run dir)"
  )
  parser.add_argument(
    "--video-length", type=int, help="play: steps recorded (default: one clip pass)"
  )
  parser.add_argument(
    "--watch",
    action="store_true",
    help="eval: also each checkpoint the training of --run-dir writes, until it ends",
  )
  parser.add_argument("--pid", type=int, help="eval --watch: the training process")
  parser.add_argument(
    "--print-cmd", action="store_true", help="print the mjlab command and exit"
  )
  args, extra = parser.parse_known_args()
  if extra and extra[0] == "--":
    extra = extra[1:]
  args.log_root = args.log_root.resolve()
  motions_dir = None if args.motions_dir is None else args.motions_dir.resolve()

  bin_dir = Path(sys.executable).parent
  if not (args.play or args.eval):
    if args.iters is None:
      parser.error("training needs --iters")
    motions_dir = motions_dir or MOTIONS_DIR
    if args.clips is None:
      motions = kept_clips(motions_dir)
    else:
      motions = sorted(motions_dir / f"{clip}.npz" for clip in args.clips.split(","))
    missing = [str(path) for path in motions if not path.is_file()]
    if not motions or missing:
      parser.error(f"no motions in {motions_dir} or missing: {missing}")
    record = library_record(motions, args.log_root)
    motion_list = library_list(motions, args.log_root)
    if not args.print_cmd:
      write_library_record(motions, record)
      write_library_list(motions, motion_list)
    # mjlab writes runs to <--log-root>/<experiment-name>/<timestamp>[_<run-name>].
    command = [
      str(bin_dir / "train"),
      TASK,
      "--env.commands.motion.motion-file",
      str(motion_list),
      "--env.scene.num-envs",
      str(args.num_envs),
      "--agent.max-iterations",
      str(args.iters),
      "--agent.save-interval",
      "4000",
      "--agent.experiment-name",
      args.log_root.name,
      "--agent.logger",
      "local",
      "--log-root",
      str(args.log_root.parent),
    ]
    if args.tag:
      command += ["--agent.run-name", args.tag]
    print(f"[train_library] {len(motions)} clips, library record {record}", flush=True)
  else:
    if args.clips is not None:
      parser.error("--clips is for training; play/eval use the run's training library")
    if args.watch and (
      not args.eval or args.run_dir is None or args.checkpoint is not None
    ):
      parser.error("--watch needs --eval and --run-dir, no --checkpoint")
    if args.pid is not None and not args.watch:
      parser.error("--pid is for --watch")
    if args.eval and args.checkpoint is None:
      run_dir = resolve_run_dir(args)
      evaluated = ["--run-dir", str(run_dir)]
    else:
      checkpoint = resolve_checkpoint(args)
      run_dir = checkpoint.parent
      evaluated = ["--checkpoint-file", str(checkpoint)]
    if args.watch and args.pid is not None and not args.print_cmd:
      wait_for_params(run_dir, args.pid)
    if motions_dir is None:
      library, motion_list = recorded_library(run_dir)
    else:
      library, motion_list = kept_clips(motions_dir), None
    if args.eval:
      if motion_list is None:
        motion_list = library_list(library, args.log_root)
        if not args.print_cmd:
          write_library_list(library, motion_list)
      command = [
        sys.executable,
        "-m",
        "mjlab.tasks.tracking.scripts.evaluate_library",
        TASK,
        *evaluated,
        "--motion-files",
        str(motion_list),
        "--num-envs",
        str(args.num_envs),
      ]
      if args.watch:
        command += ["--watch", "True"]
      if args.pid is not None:
        command += ["--pid", str(args.pid)]
    else:
      clips = {path.stem: path for path in library}
      if args.clip not in clips:
        parser.error(f"--play needs a --clip of the run's library: {sorted(clips)}")
      motion = clips[args.clip]
      video_length = args.video_length or len(np.load(motion)["joint_pos"])
      command = [
        str(bin_dir / "play"),
        TASK,
        "--checkpoint-file",
        str(checkpoint),
        "--motion-file",
        str(motion),
        "--num-envs",
        "1",
        "--viewer",
        "none",
        "--video",
        "True",
        "--video-length",
        str(video_length),
      ]
  command += extra

  # EGL device ids are physical GPU ids (EGL ignores CUDA_VISIBLE_DEVICES): without
  # MUJOCO_EGL_DEVICE_ID, video rendering lands on GPU 0.
  env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    MUJOCO_GL="egl",
    PYTHONUNBUFFERED="1",
  )
  print("[train_library]", shlex.join(command), flush=True)
  if args.print_cmd:
    return
  os.execve(command[0], command, env)


if __name__ == "__main__":
  main()
