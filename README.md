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
  policy objective, critic fit), `return_sampler.py` (the return-based reset
  distribution over 1 s bins of the library, paper App. D.1), `runner.py`
  (collection, logging, checkpoints). The hot paths run through `torch.compile` (with
  inductor's combo kernels) and CUDA graphs, the world-model fit and the policy
  update under bfloat16 autocast.
- `src/mjlab/tasks/tracking/`: a multi-clip `MotionCommand` (`motion_file` takes a
  tuple of npz or a `.txt` listing them; tracked bodies only, float16 except the
  positions), the 10-frame future-reference goal, pure tracking kernels with a
  differentiable forward kinematics (`mdp/`), the `TrackingModelInterface`
  (observation, goal, reward and termination on imagined states; `supermimic/`), the
  task `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic` (`config/g1/`) and the library
  evaluation `scripts/evaluate_library.py` (paper App. F.1: each clip once from
  frame 0, deterministic policy).
- Local logging without WandB (`src/mjlab/rl/local_log_writer.py`: TensorBoard
  events, `metrics.csv`, `curves.png`).
- `tools/`: `train_library.py` (train, evaluate or play on a clip library),
  `report_groups.py` (results per motion group, the paper's Table A.8),
  `plot_supermimic_run.py` (figures of a run), `view.py` (a policy on one clip, live
  in the browser or as a video, through `render_viser.py`), `convert_lafan.py`,
  `convert_seed.py`, `filter_seed.py` and `motion_groups.py` (the npz libraries, the
  paper's filters and motion groups), `paths.py` (where data, runs and videos live).
- Tests: `tests/test_supermimic_*.py`, `test_motion_library.py`,
  `test_tracking_kernels.py`, `test_tracking_kinematics.py`.
- `checkpoints/`: trained policies (see Trained policy).

## Install

```bash
git clone -b supermimic https://github.com/rihat99/mjlab.git mjlab-supermimic
cd mjlab-supermimic
uv sync --extra cu130            # Python 3.13, torch 2.14 for CUDA 13 (Blackwell)
PY=$PWD/.venv/bin/python
$PY -m pytest tests/
```

The commands below run from the repo root. `--extra cu126` instead for NVIDIA drivers
older than 580 (no Blackwell kernels). The GPU tools set `CUDA_VISIBLE_DEVICES` (and,
where they render, `MUJOCO_GL=egl`) from their `--gpu` argument.

## Data

Everything the tools read and write lives inside the repo, gitignored
(`tools/paths.py`); `$MJLAB_DATA`, when set, replaces `data/`, and any of these
folders may be a symlink to a larger disk:

- `data/motions/lafan/`: the 40 LAFAN1 G1 clips (`<clip>.npz`, 50 Hz).
- `data/motions/seed/`: the BONES-SEED G1 clips, `manifest.json` (which clips the
  filters keep) and `groups.json` (motion groups); `data/raw/bones_seed/`: the
  download (`g1/csv/`, `metadata/`). `data/motions/test/`: clips in neither library.
- `runs/lafan/`, `runs/seed/`: one folder `<timestamp>[_<tag>]/` per training run:
  `params/env.yaml` and `params/agent.yaml` (the full configs, with the library's
  paths), `metrics.csv` (one row per logged iteration, with `training_wall_seconds`),
  `curves.png`, TensorBoard events, `model_<it>.pt`, `eval_library_model_<it>.json`,
  `figures/`, `videos/play/`. `library_<hash>.txt` next to the runs records the
  sha256 of every clip of a training library.
- `videos/<clip>/`: videos, stills and `rollout.npz` of `tools/view.py --video`.

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
(`g1.tar.gz`, 120 fps CSVs, extracted into `data/raw/bones_seed/`). The filter writes
`manifest.json`: the paper's name filter (SONIC's keywords plus "50cm" and
"look_over_fence", except `cartwheel`, `handstand`, `handstand_ff_` and `kozak_dance`:
those clips need no prop, so we train on them) and feasibility filter. Of 142,220
clips, 129,951 are kept (264 h; the paper keeps 129,514); training uses the kept clips
when the motions dir has a manifest. `groups.json` holds the motion groups of the
paper's Table A.8 by metadata `content_name`: side rolls, fire-in-the-hole turns,
crossed legs and broad jumps with the paper's sizes, flips (54 vs 44) and lying,
resting and standing up (102 vs 99) approximate, cartwheels, handstands and the kozak
dance ours; the paper's dance routine cannot be identified by name.

```bash
$PY tools/convert_seed.py data/raw/bones_seed/g1/csv --workers 96   # ~1 min on 96 cores
$PY tools/filter_seed.py data/raw/bones_seed --workers 96           # ~30 s
$PY tools/motion_groups.py                                          # a few s
```

## Commands

Train on the LAFAN1 library (default `--motions-dir` and `--log-root`; 4096 envs;
about 0.21 s per iteration on an RTX PRO 6000) and evaluate every checkpoint as it
appears. Anything after `--` goes to mjlab's `train` unchanged, e.g. a
`SuperMimicRunnerCfg` field as `--agent.policy.horizon 16`; `--clips a,b` trains on a
subset; `--print-cmd` only prints the command. `RUN` is the run dir the training log
names first (`Logging experiment in directory: ...`); the watcher waits for its
`params/env.yaml`, loads the library once and evaluates each checkpoint until the
training process has exited:

```bash
mkdir -p runs/lafan
nohup $PY tools/train_library.py --gpu 0 --iters 14000 --tag lafan \
    -- --agent.save-interval 2000 > runs/lafan/train.log 2>&1 &
TRAIN_PID=$!
nohup $PY tools/train_library.py --eval --watch --gpu 0 --run-dir RUN --pid $TRAIN_PID \
    > runs/lafan/eval_watch.log 2>&1 &
```

Train on BONES-SEED (the library goes to mjlab as a path list; each launch hashes the
clips, ~1 min, and loads them, ~7–10 min; **TBD (SEED retrain)** s per iteration and
**TBD** GB of GPU memory at 4096 envs):

```bash
SEED="--motions-dir data/motions/seed --log-root runs/seed"
mkdir -p runs/seed
nohup $PY tools/train_library.py --gpu 0 --iters 160000 --tag seed $SEED \
    > runs/seed/train.log 2>&1 &
```

Evaluate `--checkpoint`, or every checkpoint of the run (default: the newest) that
has no `eval_library_model_<it>.json` yet (`-- --overwrite True`: all), on its
training library in one process. The JSON holds the normalised return per clip
length and per survived step, the failed clips (anchor height error over 0.25 m or
tilt over 0.8), one record per clip, the run's training time at the checkpoint and,
under `provenance`, the git commit of the evaluating code (`dirty` when tracked files
differ from it) and the robot's collision setting. The GPU simulation is not bitwise
deterministic across processes, so a clip's failure can differ between evaluations:

```bash
$PY tools/train_library.py --eval --gpu 0 [--run-dir RUN | --checkpoint RUN/model_14000.pt] [$SEED]
```

Results per motion group, as Table A.8 (from `groups.json` of the clips' folder, or
`--groups FILE`), for the last checkpoint and, over several JSONs, the return and
failed clips vs iteration and vs training hours (Fig. 3(a) and A.1), and a run's
training figures, both in `RUN/figures/`; headless play of one clip into
`RUN/videos/play/rl-video-step-0.mp4`:

```bash
$PY tools/report_groups.py RUN/eval_library_model_*.json [--groups FILE]
$PY tools/plot_supermimic_run.py RUN [--hours]   # --hours: x axis in training hours
$PY tools/train_library.py --play --gpu 0 --run-dir RUN --clip walk1_subject1
```

`tools/view.py` shows a policy on one clip live in mjlab's viser viewer (reference
ghost, pause and step, frame scrubber, the run's checkpoints to switch between; it
prints its URL, port 8080 or the next free one; from another machine run
`ssh -L 8080:localhost:8080 HOST` first). The clip comes from the run's motions dir,
else `data/motions/seed`, else `data/motions/lafan` (`--motions-dir` overrides);
anything after `--` goes to mjlab's `play` (`-- --device cpu`: no GPU). `--video`
plays the clip once as the evaluation does (printing the normalised return and
whether the policy fell) and records 20 s of it (by default the 20 s with the most
motion) as a 1920x1080, 50 fps `videos/<clip>/reference.mp4` (blue ghost = reference)
or, with `--ghost targets`, `targets.mp4` (orange ghost = joint targets), plus
`rollout.npz`; `--stills T ...` writes PNG stills instead. About 1 s per frame:

```bash
$PY tools/view.py --run-dir RUN --clip side_roll_R_001__A304_M [--gpu 0]
$PY tools/view.py --checkpoint RUN/model_14000.pt --clip dance1_subject3 \
    --video [--ghost targets] [--window-start FRAME] [--stills 2.0 4.0]
```

The video renders in a headless Chromium, by default Playwright's headless shell
(`npx playwright install chromium-headless-shell`; `--chrome PATH` for another),
which needs `libasound.so.2`; if it is missing, the tool prints how to get it, with
or without root.

## Trained policy

`checkpoints/seed100k_margin10/` was trained with the code before this branch's
method changes and no longer loads (actor input 990, now 930; critic input 831, now
840; a checkout of 12c90fb65 still evaluates it). A policy retrained with the current
code on the 129,951 kept BONES-SEED clips replaces it:

**TBD (SEED retrain): `checkpoints/<name>/model_<it>.pt`, <it> iterations (4096 envs,
<h> h on one RTX PRO 6000) at commit <sha>; normalised return <r> with <n> failed
clips on the full library.**

The folder serves as `--run-dir` for `--eval` and `--play` with
`--motions-dir data/motions/seed`, and for `tools/view.py` as it is: the run's clip
list is recorded under the training machine's paths, so the launcher takes the
library from the motions dir instead (`manifest.json`'s kept clips).

## Configuration

- `SuperMimicRunnerCfg` (`src/mjlab/rl/supermimic/config.py`): the paper's
  hyperparameters (Tables A.1–A.5); fields marked "(choice)" are not given in the
  paper. `PolicyCfg.seed_end_margin` (default 10) keeps the last state of every
  imagined rollout that many frames away from the clip end: the critic is fit on the
  first `horizon` states only, and on libraries with many clips that end mid-motion
  (SEED) a margin of 0 drove the critic loss to blow up after 1k–15k iterations.
  Critic values are clamped to `max_step_reward / (1 - gamma)` (= 10, the largest
  attainable return) in the λ-return targets, the policy's bootstrap V(s_H) and the
  reset sampler's bootstrap (`Metrics/value_clip_fraction`). Imagined actions carry
  no noise (paper Alg. A.1); `CollectionCfg.action_noise_std` acts in the simulator.
- Task (`unitree_g1_flat_tracking_supermimic_env_cfg` in
  `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`): the pelvis is the anchor for
  the fall test, the anchor rewards and the relative body targets. Full robot
  self-collision (`FULL_COLLISION`, paper App. A) without a sensor or penalty;
  friction U(0.3, 1.2) on all robot collision geoms, the ground at 0.3 in training,
  everything at 1.0 in the play config. The sim context ξ (Table A.3) is stored with
  every replay transition: friction and encoder bias are resampled at every reset,
  the torso CoM offset once per env. The actor frame is Table A.1's proprioception
  (93 per frame, 10-frame history); the goal holds the 10 reference frames (root
  position delta, 6D rotation, local root velocities, joint positions, 0.1 x joint
  velocities, in the heading frame of the current reference frame; 730); no robot
  position enters either. The critic also sees the reference root pose in the robot
  root frame (position, 6D rotation; 840 inputs in all).
- Imagined rollouts are discounted by the task's analytic soft termination (sigmoids
  of the anchor height and tilt errors; `MotionCommandCfg.termination_*_temperature`).
  Reset sampling caps a bin's softmax share at `MotionCommandCfg.softmax_cap` (50)
  times the uniform share (`Metrics/sampler_top1_probability[_uncapped]`).
- Not as in the paper: bfloat16 autocast (`PrecisionCfg.autocast`; off: fp32 with
  TF32 matmuls); the CoM offset per env, not per episode (a `body_ipos` change runs
  MuJoCo Warp's `set_const` over all worlds, +17.6 ms per env step at 4096 envs); the
  analytic termination instead of a learned one; 500 world-model-only iterations
  first (`CollectionCfg.warmup_world_model_iterations`); and our additions: the value
  clamp, the softmax cap, the critic's reference root pose and `seed_end_margin`.
