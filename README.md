# mjlab, `supermimic` branch

This branch of the mjlab fork [rihat99/mjlab](https://github.com/rihat99/mjlab)
adds SuperMimic to mjlab and trains it on 40 LAFAN1 clips with the Unitree G1.
SuperMimic is described in Liao et al., *SuperMimic: Scaling Humanoid Whole-Body
Control with One Differentiable World Model*, arXiv 2610.28480
([project page](https://supermimic.github.io)); there is no official code, so this
is a re-implementation from the paper. Upstream mjlab's README is on the `main`
branch.

## What SuperMimic is

One differentiable world model is fitted on short windows of simulator transitions
from a replay buffer, for all clips of a motion library at once. A deterministic
tracking policy is trained by backpropagation through 32-step rollouts imagined in
that world model, with a critic for the value at the end of a rollout. Resets in the
simulator favour the parts of the library where the policy earns the least return,
so data collection concentrates on what it cannot track yet.

## What this branch adds

- `src/mjlab/rl/supermimic/`: the task-agnostic learner. `config.py`
  (`SuperMimicRunnerCfg` and its parts), `interface.py` (the `ModelInterface` a task
  implements), `replay_buffer.py`, `world_model.py` (71-d floating-base state,
  root-frame inputs and increments, learned termination head), `networks.py` (motion
  encoder with a 32-d tanh latent, decoder on a 10-frame history, critic),
  `algorithm.py` (world-model fit, imagination, policy objective, critic fit),
  `runner.py` (collection, reset-sampler credit, logging, checkpoints).
- `src/mjlab/tasks/tracking/mdp/kernels.py`, `kinematics.py`: pure tracking kernels
  and a differentiable torch forward kinematics; the env terms delegate to them.
- `src/mjlab/tasks/tracking/mdp/commands.py`: multi-clip motion library
  (`motion_file` takes a tuple of npz), return-based reset sampling per clip bin,
  `motion_end` time-out; `observations.py`: 10-frame future-reference goal.
- `src/mjlab/tasks/tracking/supermimic/`: `TrackingModelInterface` (observation,
  goal, reward and termination on imagined states) and the terminal-state recorder.
- Task `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic`
  (`src/mjlab/tasks/tracking/config/g1/`).
- `src/mjlab/tasks/tracking/scripts/evaluate_library.py`: library evaluation (paper
  App. F.1: each clip once from frame 0, one env per clip, play config,
  deterministic policy).
- Shared with the other fork branches: local logging without WandB
  (`src/mjlab/rl/local_log_writer.py`: TensorBoard events, `metrics.csv`,
  `curves.png`), logs finalised on crash or SIGTERM, `csv_to_npz` writing a local
  npz.
- Tests: `tests/test_supermimic_rl.py`, `test_supermimic_interface.py`,
  `test_motion_library.py`, `test_tracking_kernels.py`,
  `test_tracking_kinematics.py`.
- `tools/` (scripts, run with the worktree's interpreter):
  - `paths.py`: the data root and its subfolders;
  - `train_library.py`: train, evaluate or play on a clip library;
  - `eval_watch.sh`: evaluates each checkpoint of a running training as it appears;
  - `plot_supermimic_run.py`: training curves and library-evaluation figures;
  - `render_viser.py`: rendering library (look, ghost model, viser scene, headless
    browser, caption);
  - `render_viser_supermimic.py`: SuperMimic rollout of one clip and its two videos;
  - `convert_lafan.py`: rebuilds `motions_lafan/` from the LAFAN1 G1 CSVs.

The additions are also listed in `docs/source/changelog.rst`.

## Install

```bash
cd mjlab-supermimic              # this branch's checkout
uv sync --extra cu130            # Python 3.13, torch 2.14 for CUDA 13 (Blackwell)
PY=$PWD/.venv/bin/python
DATA=${MJLAB_DATA:-$PWD/../retarget_test/mjlab}
```

`--extra cu126` instead for NVIDIA drivers older than 580 (no Blackwell kernels).
The tools set `CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID`, `MUJOCO_GL=egl` and
`WANDB_MODE=disabled` from their `--gpu` argument.

## Data

All data lives under one root, `$MJLAB_DATA`, by default
`<repo root>/../retarget_test/mjlab` (`tools/paths.py`). The tools use:

- `motions_lafan/<clip>.npz`: the 40 LAFAN1 G1 clips at 50 Hz (441,140 frames,
  2.45 h), with the conversion logs and a README.
- `runs/lafan_supermimic/<timestamp>[_<tag>]/`: one training run each.
  `params/env.yaml`, `params/agent.yaml` (full configs; `env.yaml` records the
  library's paths), `git/` (diff of the code it ran), `metrics.csv` (one row per
  logged iteration), `curves.png`, TensorBoard `events.out.tfevents.*`,
  `model_<it>.pt`, `eval_library_model_<it>.json` (library evaluation of that
  checkpoint), `figures/` (`plot_supermimic_run.py`), `videos/play/` (play).
  `runs/lafan_supermimic/library_<hash>.txt` records the sha256 of every clip of a
  training library (`sha256sum -c` re-checks it).
- `viser/supermimic/<clip>/`: rendered videos, stills and `rollout.npz`.

Clip npz format (read by `MotionLoader` in `mdp/commands.py`; T frames, 30 bodies of
the G1, world frame):

| key | shape | dtype |
|---|---|---|
| `fps` | (1,) | float64, 50 |
| `joint_pos`, `joint_vel` | (T, 29) | float32 |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | (T, 30, 3) | float32 |
| `body_quat_w` (wxyz) | (T, 30, 4) | float32 |

Rebuild `motions_lafan/` from the
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(`g1/*.csv`, 30 fps; motion data licence CC BY-NC-ND 4.0). The conversion takes
20–100 s per clip and skips clips whose npz exists:

```bash
$PY tools/convert_lafan.py <LAFAN1_Retargeting_Dataset>/g1 --gpu 0 \
    [--output-dir $DATA/motions_lafan] [--clips walk1_subject1,...]
```

It runs mjlab's `csv_to_npz` (30 -> 50 Hz) with one change: the per-frame velocity
self-check uses atol = rtol = 1e-4, because one frame of fallAndGetUp2_subject2
differs by 1.22e-5 (float32 default atol 1e-5) and the plain converter aborts. The
output is bitwise identical to the existing `motions_lafan/` files (checked on
fallAndGetUp3_subject1 and fallAndGetUp2_subject2).

Runs trained before the data moved record the old absolute clip paths in
`params/env.yaml`; `train_library.py` and `render_viser_supermimic.py` then take
the clips from `$MJLAB_DATA/motions_lafan/` by file name and say so.

## Commands

Train 14,000 iterations on all 40 clips (4096 envs, checkpoint every 2000; about
1.6 h on an idle RTX PRO 6000), and evaluate every checkpoint as it appears:

```bash
nohup $PY tools/train_library.py --gpu 0 --iters 14000 --tag final_learnedhead \
    -- --agent.save-interval 2000 > $DATA/runs/lafan_supermimic/train.log 2>&1 &
TRAIN_PID=$!  # the trainer itself (train_library.py execs it)
# once the run dir <timestamp>_final_learnedhead exists:
GPU=0 nohup tools/eval_watch.sh $DATA/runs/lafan_supermimic/<run> $TRAIN_PID \
    > $DATA/runs/lafan_supermimic/eval_watch.log 2>&1 &
```

The analytic-termination arm adds `--agent.policy.analytic-termination True` after
`--`. Anything after `--` goes to mjlab's `train` unchanged; `--clips a,b` trains on
a subset; `--print-cmd` only prints the command.

Evaluate a checkpoint (default: the newest run and its last checkpoint) on its
training library; writes `eval_library_model_<it>.json` next to it:

```bash
$PY tools/train_library.py --eval --gpu 0 [--run-dir RUN | --checkpoint RUN/model_14000.pt]
```

Play one clip headless and record `RUN/videos/play/rl-video-step-0.mp4` (one clip
pass by default):

```bash
$PY tools/train_library.py --play --gpu 0 --run-dir RUN --clip walk1_subject1 [--video-length 500]
```

Figures (`RUN/figures/training_curves.png`, `library_eval.png`, `per_clip_eval.png`)
and a markdown table of the checkpoint evaluations:

```bash
$PY tools/plot_supermimic_run.py RUN
```

Render a 20 s window (default: the most motion) of one clip as two 1920x1080 videos
in `$DATA/viser/supermimic/<clip>/`: `reference.mp4` (blue ghost = reference motion)
and `targets.mp4` (orange ghost = commanded joint targets):

```bash
$PY tools/render_viser_supermimic.py --gpu 0 --chrome CHROME \
    --checkpoint RUN/model_14000.pt --clip dance1_subject3 \
    [--window-start FRAME] [--stills 55 --only-stills]
```

`CHROME` is a Chromium headless shell or a wrapper that execs one with
`--no-sandbox`. Get one with `npx playwright install chromium-headless-shell` (it
lands in `~/.cache/ms-playwright/chromium_headless_shell-<version>/`). On a machine
without `libasound.so.2`, extract it from the distribution package (`apt-get download
libasound2t64; dpkg-deb -x libasound2t64_*.deb asound`) and let the wrapper put
`asound/usr/lib/x86_64-linux-gnu` on `LD_LIBRARY_PATH`:

```bash
#!/bin/bash
export LD_LIBRARY_PATH=<asound>/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec ~/.cache/ms-playwright/chromium_headless_shell-<version>/chrome-headless-shell-linux64/chrome-headless-shell --no-sandbox "$@"
```

## Results

14,000-iteration runs on the 40 clips, 4096 envs, default config (paper-faithful
learned termination head). sim = training normalised return in the simulator;
imagined = imagined reward normalised by its maximum; eval = library evaluation of
the checkpoint, normalised by clip length ("clip", penalises an early fall) and by
survived steps ("survived").

| iteration | sim | imagined | eval clip | eval survived | failed clips |
|---|---|---|---|---|---|
| 2000 | 0.582 | 0.590 | 0.495 | 0.655 | 17 / 40 |
| 6000 | 0.782 | 0.792 | 0.791 | 0.839 | 6 / 40 |
| 10000 | 0.819 | 0.825 | 0.883 | 0.888 | 3 / 40 |
| 14000 | 0.843 | 0.861 | 0.855 | 0.907 | 6 / 40 |

A colleague's independent implementation (Isaac Lab / Newton, same clips and
hyperparameters) reaches 0.842 survived with 0 / 40 failed at 14,000 iterations. The
analytic-termination arm reaches 0.884 clip / 0.899 survived / 2 failed. The two
evaluators use different simulators and reward implementations, so the offset only
shows the same level, not a better policy. Speed: 0.40 s per iteration on an idle
RTX PRO 6000 (paper: 0.11 s on an H100).

Runs: `$DATA/runs/lafan_supermimic/2026-10-23_00-33-44_final_learnedhead/` and
`2026-10-23_00-33-45_final_analyticterm/`. Detailed report (curves, per-clip table,
the compile bug, speed, comparison with the colleague's run):
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-22_14-35_supermimic/REPORT.md`.

## Configuration pointers

- `SuperMimicRunnerCfg` (`src/mjlab/rl/supermimic/config.py`): every paper
  hyperparameter (Tables A.1–A.5); fields marked "(choice)" are not given in the
  paper. Override any field from the command line after `--`, e.g.
  `--agent.policy.horizon 16`.
- `PolicyCfg.analytic_termination`: imagination uses the task's analytic soft
  termination instead of the learned head (the head keeps training, for logging).
- `PrecisionCfg.compile` (default on): `torch.compile` plus CUDA graphs for the
  world-model loss, the imagined step, the state heads and the critic loss. The
  policy objective always runs eagerly: compiled, its forward matched eager but its
  backward did not (gradient cosine 0.52–0.54 to the eager gradient), and every
  run trained with it plateaued at a normalised return of 0.40. Eager costs about
  14 % per iteration. `--agent.precision.compile False` runs everything eagerly.

## Known limitations and open items

- The exact operation `torch.compile` mis-differentiates in the policy objective is
  not localised; re-enabling it needs a gradient test with the joint-velocity reward
  path (the synthetic-task test in `tests/test_supermimic_rl.py` does not exercise
  it).
- Ablation suggested by the gradient diagnostic, not run: the critic-bootstrap term
  alone collapsed the policy, reward-only improved more than the full objective over
  25 updates.
- Untested design decisions (only confounded runs): pelvis vs torso anchor, raw-metre
  goal scaling, critic error features.
- Training episodes terminate about ten times more often than in the colleague's
  run; evaluation failures are single rollouts per clip and change between
  checkpoints (2–7). Align the fall criterion and average a few rollouts per clip
  before the next comparison.
- Paper budget not run: 320k iterations, about 36 h at 0.40 s.
- BONES-SEED dataset: not started.
- `render_viser_supermimic.py`: the second video rendered in the same process is
  3–10x slower than the first; render one ghost per process (`--ghosts reference`,
  then `--ghosts targets`) when time matters.
- `tests/test_tracking_kernels.py::test_recorded_rollout_parity` and
  `tests/test_tracking_kinematics.py::test_fk_matches_recorded_rollout` need
  rollouts recorded before the kernel refactor by a `record_baseline.py` that is not
  in this branch; they skip without `TRACKING_PARITY_BEFORE` /
  `TRACKING_PARITY_AFTER`.
