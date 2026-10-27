"""View a SuperMimic policy on one clip: live in the browser (viser), or as a video.

Live (default): mjlab's ``play`` with its viser viewer, 1 env on the clip: the robot
with the reference motion as a ghost, pause, a frame scrubber, and the checkpoints of
the run dir to switch between. viser prints its URL (port 8080, or the next free
one); from another machine, forward the port first (``ssh -L 8080:localhost:8080
HOST``). Ctrl-C stops it. Anything after ``--`` goes to ``play`` unchanged, e.g.
``-- --device cpu``. ``checkpoints/seed`` is the shipped BONES-SEED policy
(iteration 144000):

    PY=<repo>/.venv/bin/python
    $PY tools/view.py --run-dir checkpoints/seed --clip side_roll_R_001__A304_M
    $PY tools/view.py --checkpoint runs/lafan/<run>/model_14000.pt --clip dance1_subject3

Video (``--video``): one deterministic rollout of the whole clip from frame 0, as the
library evaluation plays it (prints the normalised return and whether it failed),
then a 1920x1080, 50 fps video of 20 s of it (all of it if shorter, up to a fall; by
default the 20 s with the most root motion, else from ``--window-start``), rendered
through viser in a headless Chromium on ``--gpu``. Writes ``videos/<clip>/``:

- ``reference.mp4`` (``--ghost reference``, the default): blue ghost = the reference
  motion;
- ``targets.mp4`` (``--ghost targets``): orange ghost = the commanded joint targets;
- ``<ghost>_t<T>.png``: with ``--stills T ...`` (clip times, s), instead of the video;
- ``rollout.npz``: the whole rollout (per 200 Hz substep ``qpos``, ``ctrl``; per
  policy step ``frame``, ``reward``, ``ref_qpos``) and the window.

    $PY tools/view.py --run-dir checkpoints/seed \\
        --clip side_roll_R_001__A304_M --video [--ghost targets] [--window-start 100]

The browser is Playwright's Chromium headless shell (``npx playwright install
chromium-headless-shell``) unless ``--chrome`` names another; it needs
``libasound.so.2`` (the tool prints how to get it if missing).

The clip is ``<motions dir>/<clip>.npz``. Without ``--motions-dir`` it is the first of
these folders that has the clip: the one the run's ``params/env.yaml`` records,
``data/motions/seed``, ``data/motions/lafan`` (``$MJLAB_DATA`` replaces ``data``;
``tools/paths.py``).
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import yaml
from paths import MOTIONS_DIR, MOTIONS_SEED_DIR, VIDEOS_DIR
from render_viser import (
  GHOSTS,
  WINDOW_S,
  best_window,
  check_chrome,
  find_chrome,
  free_port,
  render_video,
  rollout,
  window_states,
)
from train_library import TASK, ParamsLoader, latest_checkpoint


def recorded_motions_dir(run_dir: Path) -> Path | None:
  """Folder of the first library file that ``run_dir/params/env.yaml`` records.

  None without the record, or if its ``.txt`` clip list is not on this machine.
  """
  params = run_dir / "params" / "env.yaml"
  if not params.is_file():
    return None
  env_cfg = yaml.load(params.read_text(), Loader=ParamsLoader)
  recorded = env_cfg["commands"]["motion"]["motion_file"]
  first = recorded if isinstance(recorded, str) else recorded[0]
  if first.endswith(".txt"):
    if not Path(first).is_file():
      return None
    with open(first) as listing:
      first = listing.readline().strip()
  return Path(first).parent


def resolve_motion(clip: str, motions_dir: Path | None, run_dir: Path) -> Path:
  """``<motions_dir>/<clip>.npz``.

  Without ``motions_dir``: the first of the run's recorded motions dir,
  MOTIONS_SEED_DIR and MOTIONS_DIR that has the clip.
  """
  if motions_dir is not None:
    folders = [motions_dir]
  else:
    recorded = recorded_motions_dir(run_dir)
    folders = [MOTIONS_SEED_DIR, MOTIONS_DIR]
    if recorded is not None:
      folders.insert(0, recorded)
  for folder in folders:
    if (folder / f"{clip}.npz").is_file():
      return folder / f"{clip}.npz"
  raise SystemExit(
    f"no {clip}.npz in {', '.join(map(str, folders))} (choose --motions-dir)"
  )


def live(checkpoint: Path, motion: Path, gpu: int, extra: list[str]) -> None:
  """Replace this process with mjlab's ``play`` and its viser viewer."""
  command = [
    str(Path(sys.executable).parent / "play"),
    TASK,
    "--checkpoint-file",
    str(checkpoint),
    "--motion-file",
    str(motion),
    "--num-envs",
    "1",
    "--viewer",
    "viser",
    *extra,
  ]
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
  print("[view]", shlex.join(command), flush=True)
  os.execve(command[0], command, env)


def record(args: argparse.Namespace, checkpoint: Path, motion: Path) -> None:
  """Roll out ``motion`` once and render the window to ``VIDEOS_DIR/<clip>/``."""
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before the first CUDA call
  import viser

  chrome = args.chrome or find_chrome()
  check_chrome(chrome)
  states = rollout(TASK, checkpoint, motion)
  print(
    f"[view] {len(states['step_reward'])} steps, failed {states['failed']}, "
    f"normalised return {states['normalized_return']:.4f}",
    flush=True,
  )
  # A clip of F frames scores frames 1 .. F - 1; a fall ends the rollout earlier.
  last_scored = int(states["step_frame"][-1])
  num_steps = min(round(WINDOW_S / states["step_dt"]), last_scored)
  first_frame = args.window_start
  if first_frame is None:
    first_frame = best_window(motion, num_steps, last_scored)
  window = window_states(states, first_frame, num_steps)
  last_frame = first_frame + num_steps - 1
  print(
    f"[view] window: scored frames {first_frame} .. {last_frame} "
    f"(t = {first_frame * states['step_dt']:.2f} .. "
    f"{last_frame * states['step_dt']:.2f} s)",
    flush=True,
  )
  out_dir = VIDEOS_DIR / args.clip
  out_dir.mkdir(parents=True, exist_ok=True)
  np.savez(
    out_dir / "rollout.npz",
    qpos=states["sub_qpos"],
    ctrl=states["sub_ctrl"],
    frame=states["step_frame"],
    reward=states["step_reward"],
    ref_qpos=states["step_ref_qpos"],
    failed=states["failed"],
    normalized_return=states["normalized_return"],
    window_frames=(first_frame, num_steps),
    physics_dt=states["physics_dt"],
    step_dt=states["step_dt"],
    checkpoint=str(checkpoint),
    motion=str(motion),
  )
  server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
  try:
    render_video(
      args.clip,
      window,
      server,
      out_dir,
      args.ghost or "reference",
      chrome,
      None if args.software else args.gpu,
      args.stills or [],
    )
  finally:
    server.stop()
  print(f"[view] wrote {out_dir}", flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint", type=Path, help="checkpoint file model_<it>.pt")
  source.add_argument("--run-dir", type=Path, help="run dir: its last checkpoint")
  parser.add_argument("--clip", required=True, help="clip name (npz file stem)")
  parser.add_argument(
    "--motions-dir", type=Path, help="folder of the clip (default: see above)"
  )
  parser.add_argument(
    "--gpu", type=int, default=0, help="GPU of the simulation and the video's WebGL"
  )
  parser.add_argument(
    "--video",
    action="store_true",
    help="record videos/<clip>/ instead of the live viewer",
  )
  video = parser.add_argument_group("video options")
  video.add_argument(
    "--ghost", choices=tuple(GHOSTS), help="ghost of the video (default: reference)"
  )
  video.add_argument(
    "--window-start",
    type=int,
    help="first scored frame of the 20 s window (default: most root motion)",
  )
  video.add_argument(
    "--stills", type=float, nargs="+", help="clip times (s): PNG stills, no video"
  )
  video.add_argument(
    "--chrome", help="Chromium executable (default: Playwright's headless shell)"
  )
  video.add_argument(
    "--software", action="store_true", help="WebGL in SwiftShader instead of the GPU"
  )
  argv = sys.argv[1:]
  split = argv.index("--") if "--" in argv else len(argv)
  args, extra = parser.parse_args(argv[:split]), argv[split + 1 :]
  video_only = (args.ghost, args.window_start, args.stills, args.chrome)
  if not args.video and (any(v is not None for v in video_only) or args.software):
    parser.error("--ghost, --window-start, --stills, --chrome, --software need --video")
  if args.video and extra:
    parser.error("arguments after -- are for the live viewer")

  checkpoint = args.checkpoint or latest_checkpoint(args.run_dir)
  if not checkpoint.is_file():
    parser.error(f"no checkpoint {checkpoint}")
  checkpoint = checkpoint.absolute()
  motion = resolve_motion(args.clip, args.motions_dir, checkpoint.parent).absolute()
  print(f"[view] {checkpoint} on {motion}", flush=True)
  if args.video:
    record(args, checkpoint, motion)
  else:
    live(checkpoint, motion, args.gpu, extra)


if __name__ == "__main__":
  main()
