#!/usr/bin/env python3
"""Train, play or evaluate one mjlab (BeyondMimic) motion tracker for one clip, fully
local (no WandB).

Train: the motion is <motions-dir>/<clip>.npz (from convert_motion.py); the run dir is
<log-root>/<clip>/<timestamp>[_<tag>]/ and holds params/, TensorBoard events,
metrics.csv, curves.png, model_<it>.pt and <run dir name>.onnx. The run trains on a
read-only copy <log-root>/<clip>/motion_<sha256[:12]>.npz of the motion, and
params/env.yaml records that path, so re-converting the clip later never changes what an
old run is played or evaluated on:

    PY=.venv/bin/python  # from the repo root
    $PY tools/train_tracker.py --clip amass_cmu_88_06 --gpu 0 --iters 30000 \\
        --tag full [--task Mjlab-Tracking-Flat-Unitree-G1-Vic-Freq]

Play headless (no viewer): records <run dir>/videos/play/rl-video-step-0.mp4, one
motion pass by default, then exits. Evaluate: upstream tracking metrics (plus the VIC
smoothness and gain metrics) -> <run dir>/eval_model_<it>.json. Both default to the
newest run of the clip and its last checkpoint; the motion and the task id are the ones
recorded in the run's params/env.yaml (an explicit --task that differs from the run's is
an error):

    $PY tools/train_tracker.py --play --clip amass_cmu_88_06 --gpu 0 \\
        [--run-dir DIR | --checkpoint FILE] [--video-length N]
    $PY tools/train_tracker.py --eval --clip amass_cmu_88_06 --gpu 0 \\
        [--run-dir DIR | --checkpoint FILE] [--num-envs 1024]

Anything after ``--`` is passed to mjlab's train / play / evaluate unchanged (e.g.
``-- --agent.save-interval 100`` or ``-- --freeze-gains True``).

``--mps``: several trainings on one GPU only run faster than one when they share it
through CUDA MPS (without it the GPU time-slices them). One user-level daemon per GPU,
started and stopped with:

    D=$MJLAB_DATA/mps/gpu0; mkdir -p $D/pipe $D/log
    CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=$D/pipe \\
        CUDA_MPS_LOG_DIRECTORY=$D/log nvidia-cuda-mps-control -d
    echo quit | CUDA_MPS_PIPE_DIRECTORY=$D/pipe nvidia-cuda-mps-control

The mjlab process replaces this one (exec), so a nohup'ed PID is the trainer's PID.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import yaml
from paths import MOTIONS_DIR, MPS_DIR, RUNS_DIR, VENV_BIN

TASK = "Mjlab-Tracking-Flat-Unitree-G1"


class ParamsLoader(yaml.SafeLoader):
  """Reads mjlab's params/env.yaml as plain data; its python/* tagged nodes
  (functions, enums, tuples) become None."""


ParamsLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def latest_checkpoint(run_dir: Path) -> Path:
  """``model_<it>.pt`` with the highest ``it`` in ``run_dir``."""
  checkpoints = [
    p for p in run_dir.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)
  ]
  if not checkpoints:
    raise FileNotFoundError(f"no model_<it>.pt in {run_dir}")
  return max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))


def frozen_motion_copy(motion: Path, clip_dir: Path) -> Path:
  """Read-only, content-addressed copy ``<clip_dir>/motion_<sha256[:12]>.npz`` of
  ``motion``."""
  data = motion.read_bytes()
  target = clip_dir / f"motion_{hashlib.sha256(data).hexdigest()[:12]}.npz"
  if not target.exists():
    clip_dir.mkdir(parents=True, exist_ok=True)
    # Unique per process: concurrent runs of one clip write the same copy.
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.chmod(0o444)
    os.replace(tmp_path, target)
  return target


def run_motion_file(env_cfg: dict, run_dir: Path) -> Path:
  """Motion file a run trained on: the path in its params/env.yaml or, when the data
  root has moved since, the file of that name next to the run dir (where
  ``frozen_motion_copy`` put it)."""
  recorded = Path(env_cfg["commands"]["motion"]["motion_file"])
  if recorded.is_file():
    return recorded
  moved = run_dir.parent / recorded.name
  if not moved.is_file():
    raise FileNotFoundError(f"motion of {run_dir} not found: {recorded} nor {moved}")
  return moved


def run_task_id(env_cfg: dict) -> str:
  """Task id a run was trained on, read off the action, reward and observation terms
  of its params/env.yaml."""
  gain = env_cfg["actions"].get("joint_gain")
  if gain is not None:
    return TASK + {"freq": "-Vic-Freq", "freq_damp": "-Vic-FreqDamp"}[gain["mode"]]
  if "target_tracking_l2" in env_cfg["rewards"]:
    return TASK + "-NewRew"
  if "base_lin_vel" not in env_cfg["observations"]["actor"]["terms"]:
    return TASK + "-No-State-Estimation"
  return TASK


def resolve_checkpoint(args: argparse.Namespace) -> Path:
  if args.checkpoint is not None:
    return args.checkpoint
  run_dir = args.run_dir
  if run_dir is None:
    if args.clip is None:
      raise SystemExit("--play/--eval need --clip, --run-dir or --checkpoint")
    # Run dir names start with their timestamp.
    runs = sorted(p for p in (args.log_root / args.clip).iterdir() if p.is_dir())
    if not runs:
      raise FileNotFoundError(f"no runs under {args.log_root / args.clip}")
    run_dir = runs[-1]
  return latest_checkpoint(run_dir)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
    "--play", action="store_true", help="headless video of a trained checkpoint"
  )
  mode.add_argument(
    "--eval", action="store_true", help="tracking metrics JSON of a trained checkpoint"
  )
  parser.add_argument(
    "--clip",
    help="clip id (motion <motions-dir>/<clip>.npz, runs under <log-root>/<clip>/)",
  )
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--iters", type=int, help="training iterations (training only)")
  parser.add_argument(
    "--num-envs", type=int, help="default: 4096 train, 1024 eval, 1 play"
  )
  parser.add_argument("--tag", default="", help="suffix of the run dir name")
  parser.add_argument(
    "--task",
    help=f"mjlab task id (default: {TASK} for training, the run's for play/eval)",
  )
  parser.add_argument(
    "--motion-file",
    type=Path,
    help="default: <motions-dir>/<clip>.npz (train), the run's (play/eval)",
  )
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--log-root", type=Path, default=RUNS_DIR)
  parser.add_argument(
    "--run-dir", type=Path, help="play/eval: run dir (default: newest run of --clip)"
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    help="play/eval: checkpoint file (default: last in run dir)",
  )
  parser.add_argument(
    "--video-length", type=int, help="play: steps recorded (default: one motion pass)"
  )
  parser.add_argument(
    "--mps",
    action="store_true",
    help="run as a client of the MPS daemon of --gpu (<data root>/mps/gpu<N>)",
  )
  args, extra = parser.parse_known_args()
  if extra and extra[0] == "--":
    extra = extra[1:]

  if not (args.play or args.eval):
    if args.clip is None or args.iters is None:
      parser.error("training needs --clip and --iters")
    motion = (args.motion_file or args.motions_dir / f"{args.clip}.npz").resolve()
    if not motion.is_file():
      parser.error(f"motion file not found: {motion} (run convert_motion.py first)")
    motion = frozen_motion_copy(motion, args.log_root.resolve() / args.clip)
    command = [
      str(VENV_BIN / "train"), args.task or TASK,
      "--env.commands.motion.motion-file", str(motion),
      "--env.scene.num-envs", str(args.num_envs or 4096),
      "--agent.max-iterations", str(args.iters),
      "--agent.experiment-name", args.clip,
      "--agent.logger", "local",
      "--log-root", str(args.log_root.resolve()),
    ]  # fmt: skip
    if args.tag:
      command += ["--agent.run-name", args.tag]
  else:
    checkpoint = resolve_checkpoint(args).resolve()
    params_dir = checkpoint.parent / "params"
    if args.clip is not None:
      agent_cfg = yaml.load(
        (params_dir / "agent.yaml").read_text(), Loader=ParamsLoader
      )
      if agent_cfg["experiment_name"] != args.clip:
        parser.error(
          f"--clip {args.clip}, but {checkpoint} was trained on "
          f"{agent_cfg['experiment_name']}"
        )
    env_cfg = yaml.load((params_dir / "env.yaml").read_text(), Loader=ParamsLoader)
    task = run_task_id(env_cfg)
    if args.task is not None and args.task != task:
      parser.error(f"--task {args.task}, but {checkpoint} was trained on {task}")
    motion = args.motion_file or run_motion_file(env_cfg, checkpoint.parent)
    if args.play:
      video_length = args.video_length or len(np.load(motion)["joint_pos"])
      command = [
        str(VENV_BIN / "play"), task,
        "--checkpoint-file", str(checkpoint), "--motion-file", str(motion),
        "--num-envs", str(args.num_envs or 1),
        "--viewer", "none", "--video", "True", "--video-length", str(video_length),
      ]  # fmt: skip
    else:
      command = [
        str(VENV_BIN / "python"), "-m", "mjlab.tasks.tracking.scripts.evaluate", task,
        "--checkpoint-file", str(checkpoint), "--motion-file", str(motion),
        "--num-envs", str(args.num_envs or 1024),
      ]  # fmt: skip
  command += extra

  # EGL device ids are physical GPU ids (EGL ignores CUDA_VISIBLE_DEVICES): without
  # MUJOCO_EGL_DEVICE_ID, video rendering lands on GPU 0.
  env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    WANDB_MODE="disabled",
    MUJOCO_GL="egl",
  )
  if args.mps:
    # The daemon was started with CUDA_VISIBLE_DEVICES=<gpu>; its clients number
    # devices relative to it.
    mps_dir = MPS_DIR / f"gpu{args.gpu}"
    if not (mps_dir / "pipe" / "control").exists():
      parser.error(f"no MPS daemon socket in {mps_dir}/pipe")
    env.update(
      CUDA_VISIBLE_DEVICES="0",
      CUDA_MPS_PIPE_DIRECTORY=str(mps_dir / "pipe"),
      CUDA_MPS_LOG_DIRECTORY=str(mps_dir / "log"),
    )
  print("[train_tracker]", " ".join(command), flush=True)
  os.execve(command[0], command, env)


if __name__ == "__main__":
  main()
