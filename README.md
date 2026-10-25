# mjlab, `supermimic` branch

This branch of the mjlab fork [rihat99/mjlab](https://github.com/rihat99/mjlab)
adds SuperMimic, model-based whole-body motion tracking over a clip library, for the
Unitree G1. SuperMimic is described in Liao et al., *SuperMimic: Scaling Humanoid
Whole-Body Control with One Differentiable World Model*, arXiv 2610.28480
([project page](https://supermimic.github.io)); there is no official code, so this is
a re-implementation from the paper. Upstream mjlab's README is on the `main` branch.

## What SuperMimic is

One differentiable world model is fitted on short windows of simulator transitions
from a replay buffer, for all clips of a motion library at once. A deterministic
tracking policy is trained by backpropagation through 32-step rollouts imagined in
that world model, with a critic for the value at the end of a rollout. Resets in the
simulator favour the parts of the library where the policy earns the least return,
so data collection concentrates on what it cannot track yet.

## What this branch adds

- `src/mjlab/rl/supermimic/`: the task-agnostic learner. `config.py`
  (`SuperMimicRunnerCfg`, every paper hyperparameter), `interface.py` (the
  `ModelInterface` a task implements), `replay_buffer.py`, `world_model.py`,
  `networks.py` (policy and critic), `algorithm.py` (world-model fit, imagination,
  policy objective, critic fit), `runner.py` (collection, reset sampling, logging,
  checkpoints). The hot paths run through `torch.compile` and CUDA graphs, the
  world-model fit and the policy update under bfloat16 autocast.
- `src/mjlab/tasks/tracking/`: a multi-clip `MotionCommand` (`motion_file` takes a
  tuple of npz or a `.txt` listing them; only the tracked bodies are kept, in float16
  except the body positions), the 10-frame future-reference goal, pure tracking
  kernels with a differentiable forward kinematics (`mdp/kernels.py`,
  `mdp/kinematics.py`), the `TrackingModelInterface` that evaluates observation,
  goal, reward and termination on imagined states (`supermimic/`), the task
  `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic` (`config/g1/`) and the library
  evaluation `scripts/evaluate_library.py` (paper App. F.1: each clip once from
  frame 0, deterministic policy, in waves of at most `--num-envs` clips).
- Local logging without WandB (`src/mjlab/rl/local_log_writer.py`: TensorBoard
  events, `metrics.csv`, `curves.png`).
- `tools/`: `train_library.py` (train, evaluate or play on a clip library),
  `eval_watch.sh` (evaluates each checkpoint of a running training as it appears),
  `plot_supermimic_run.py` (training curves and evaluation figures of a run),
  `render_viser_supermimic.py` with `render_viser.py` (videos of a rollout),
  `convert_lafan.py` (LAFAN1 G1 CSVs to the npz library), `convert_seed.py` and
  `filter_seed.py` (BONES-SEED to the npz library with the paper's filters),
  `paths.py` (the data root).
- Tests: `tests/test_supermimic_rl.py`, `test_supermimic_interface.py`,
  `test_motion_library.py`, `test_tracking_kernels.py`, `test_tracking_kinematics.py`.
- A trained policy, `checkpoints/seed100k_margin10/` (see Trained policy).

## Install

```bash
git clone -b supermimic https://github.com/rihat99/mjlab.git mjlab-supermimic
cd mjlab-supermimic
uv sync --extra cu130            # Python 3.13, torch 2.14 for CUDA 13 (Blackwell)
PY=$PWD/.venv/bin/python
DATA=${MJLAB_DATA:-$PWD/../retarget_test/mjlab}
$PY -m pytest tests/
```

`--extra cu126` instead for NVIDIA drivers older than 580 (no Blackwell kernels).
The GPU tools set `CUDA_VISIBLE_DEVICES` (and, where they render, `MUJOCO_GL=egl`)
from their `--gpu` argument.

## Data

All data lives under one root, `$MJLAB_DATA`, by default
`<repo root>/../retarget_test/mjlab` (`tools/paths.py`):

- `motions_lafan/<clip>.npz`: the 40 LAFAN1 G1 clips at 50 Hz.
- `motions_seed/<clip>.npz`: the BONES-SEED G1 clips at 50 Hz and `manifest.json`
  (which clips the filters keep).
- `runs/<library>_supermimic/<timestamp>[_<tag>]/`: one training run each:
  `params/env.yaml` and `params/agent.yaml` (the full configs; `env.yaml` records the
  library's paths), `metrics.csv`, `curves.png`, TensorBoard events, `model_<it>.pt`,
  `eval_library_model_<it>.json`, `figures/`, `videos/play/`. `library_<hash>.txt`
  next to the runs records the sha256 of every clip of a training library.
- `viser/supermimic/<clip>/`: rendered videos, stills and `rollout.npz`.

Clip npz format (T frames, the 30 bodies of the G1, world frame):

| key | shape | dtype |
|---|---|---|
| `fps` | (1,) | float64, 50 |
| `joint_pos`, `joint_vel` | (T, 29) | float32 |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | (T, 30, 3) | float32 |
| `body_quat_w` (wxyz) | (T, 30, 4) | float32 |

LAFAN1, from the
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(`g1/*.csv`, 30 fps; motion data licence CC BY-NC-ND 4.0); mjlab's `csv_to_npz`
resamples to 50 Hz, 20–100 s per clip:

```bash
$PY tools/convert_lafan.py <LAFAN1_Retargeting_Dataset>/g1 --gpu 0 [--clips walk1_subject1,...]
```

BONES-SEED, from [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)
(`g1.tar.gz`, 120 fps CSVs). The filter writes `manifest.json`: the paper's name filter
(SONIC's keywords plus "50cm" and "look_over_fence") and feasibility filter, and a
third step that drops clips in which a joint moves more than 1 rad between two frames
(retargeting glitches). Of 142,220 clips, 129,043 are kept (262 h); training uses the
kept clips when the motions dir has a manifest.

```bash
$PY tools/convert_seed.py <seed>/g1/csv --workers 96   # ~1 min on 96 cores
$PY tools/filter_seed.py <seed> --workers 96           # ~1 min
```

## Commands

Train on the LAFAN1 library (default `--motions-dir` and `--log-root`; 4096 envs;
about 0.25 s per iteration on an RTX PRO 6000) and evaluate every checkpoint as it
appears. Anything after `--` goes to mjlab's `train` unchanged, e.g. a
`SuperMimicRunnerCfg` field as `--agent.policy.horizon 16`; `--clips a,b` trains on a
subset; `--print-cmd` only prints the command:

```bash
nohup $PY tools/train_library.py --gpu 0 --iters 14000 --tag lafan \
    -- --agent.save-interval 2000 > $DATA/runs/lafan_supermimic/train.log 2>&1 &
TRAIN_PID=$!
GPU=0 nohup tools/eval_watch.sh $DATA/runs/lafan_supermimic/<run> $TRAIN_PID \
    > $DATA/runs/lafan_supermimic/eval_watch.log 2>&1 &
```

Train on BONES-SEED (the library goes to mjlab as a path list; each launch hashes the
clips, ~1 min, and loads them, ~7–10 min; 0.26 s per iteration and 42 GB of GPU memory
at 4096 envs):

```bash
SEED="--motions-dir $DATA/motions_seed --log-root $DATA/runs/seed_supermimic"
nohup $PY tools/train_library.py --gpu 0 --iters 320000 --tag seed $SEED \
    > $DATA/runs/seed_supermimic/train.log 2>&1 &
```

Evaluate a checkpoint (default: the newest run and its last checkpoint) on its
training library; writes `eval_library_model_<it>.json` next to it, with the
normalised return per clip length and per survived step, the failed clips (anchor
height error over 0.25 m or tilt over 0.8) and one record per clip. The GPU
simulation is not bitwise deterministic across processes, so a clip's failure can
differ between two evaluations of the same checkpoint:

```bash
$PY tools/train_library.py --eval --gpu 0 [--run-dir RUN | --checkpoint RUN/model_14000.pt] [$SEED]
```

Play one clip headless into `RUN/videos/play/rl-video-step-0.mp4`, and draw a run's
figures (`RUN/figures/`):

```bash
$PY tools/train_library.py --play --gpu 0 --run-dir RUN --clip walk1_subject1
$PY tools/plot_supermimic_run.py RUN
```

Render one clip as two 1920x1080 videos in `$DATA/viser/supermimic/<clip>/`:
`reference.mp4` (blue ghost = reference motion) and `targets.mp4` (orange ghost =
commanded joint targets); the whole clip up to 20 s, by default the 20 s with the
most motion:

```bash
$PY tools/render_viser_supermimic.py --gpu 0 --chrome CHROME \
    --checkpoint RUN/model_14000.pt --clip dance1_subject3 \
    [--window-start FRAME] [--ghosts reference] [--stills 55 --only-stills]
```

`CHROME` is a Chromium headless shell or a wrapper that execs one with
`--no-sandbox` (`npx playwright install chromium-headless-shell` puts one in
`~/.cache/ms-playwright/`). On a machine without `libasound.so.2`, extract it from
the distribution package (`apt-get download libasound2t64; dpkg-deb -x ...`) and let
the wrapper put its `usr/lib/x86_64-linux-gnu` on `LD_LIBRARY_PATH`. Render one ghost
per process when time matters: the second video of a process renders several times
slower.

## Trained policy

`checkpoints/seed100k_margin10/model_100000.pt`: 100,000 iterations on the 129,043
kept BONES-SEED clips (4096 envs, 7.3 h on one RTX PRO 6000) at commit 341536c87, the
code before the cleanup that followed it; its `params/` are the run's own record.
That code also trained a small termination head on the world model's features (about
1 % of the world-model gradient), which is gone, so a new run with the current code
is not expected to reproduce the checkpoint bit for bit. On the full library it
reaches a normalised return of 0.9605 with 4 failed clips (the paper's single-GPU
run at the same iteration: about 0.94 and 25 failed clips; SONIC: 0.90 and 179).
The folder serves as `--run-dir` for `--eval`, `--play` and the renderer with
`--motions-dir $DATA/motions_seed`: the run's clip list is recorded under this
machine's paths, so the launcher takes the library from the motions dir instead
(`manifest.json`'s kept clips). The file holds the policy, critic, world model and
reset-sampler state but no optimizer states, so it cannot resume training.

## Configuration

- `SuperMimicRunnerCfg` (`src/mjlab/rl/supermimic/config.py`): the paper's
  hyperparameters (Tables A.1–A.5); fields marked "(choice)" are not given in the
  paper. `PolicyCfg.seed_end_margin` (default 10) keeps the last state of every
  imagined rollout that many frames away from the clip end: the critic is fit on the
  first `horizon` states only, and on libraries with many clips that end mid-motion
  (SEED) a margin of 0 drove the critic loss to blow up after 1k–15k iterations.
  `PrecisionCfg.autocast` (default on) runs the world-model fit and the policy update
  under bfloat16; off keeps fp32 with TF32 matmuls.
- Task (`unitree_g1_flat_tracking_supermimic_env_cfg` in
  `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`): the pelvis is the anchor for
  the fall test, the anchor rewards and the relative body targets. Contact physics as
  in Uni-SuperMimic: no robot self-collision; the startup friction U(0.3, 1.2), one
  value per env, on all robot collision geoms, with the ground at 0.3 in training and
  everything at 1.0 in the play config. The policy input is Uni-SuperMimic's
  deployable layout: the goal holds the 10 reference frames (root position delta, 6D
  rotation, local root velocities, joint positions, 0.1 x joint velocities, in the
  heading frame of the current reference frame; 730), no robot position enters it,
  and the actor frame carries the 6D orientation error of the robot root to the
  current reference frame (99 per frame, 10-frame history).
- Imagined rollouts are discounted by the task's analytic soft termination; the
  last `horizon + seed_end_margin` frames of a clip never seed one.
