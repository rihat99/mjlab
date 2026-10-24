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
  (`motion_file` takes a tuple of npz or a `.txt` file listing them; only the tracked
  bodies are kept, in float16 except the body positions), return-based reset sampling
  per clip bin, `motion_end` time-out; `observations.py`: the 10-frame
  future-reference goal in its two layouts (see Configuration pointers) and the
  orientation-error term.
- `src/mjlab/tasks/tracking/supermimic/`: `TrackingModelInterface` (observation,
  goal, reward and termination on imagined states) and the terminal-state recorder.
- Task `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic`
  (`src/mjlab/tasks/tracking/config/g1/`); the `-AnchorGoal` variants keep the
  observation layout of the checkpoints before 2026-10-24.
- `src/mjlab/tasks/tracking/scripts/evaluate_library.py`: library evaluation (paper
  App. F.1: each clip once from frame 0, play config, deterministic policy), in waves
  of at most `--num-envs` clips (default 4096), longest clips first.
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
  - `convert_lafan.py`: rebuilds `motions_lafan/` from the LAFAN1 G1 CSVs;
  - `convert_seed.py`, `filter_seed.py`: build `motions_seed/` from BONES-SEED and
    apply the paper's App. E filters.

The additions are also listed in `docs/source/changelog.rst`.

## Install

```bash
git clone -b supermimic https://github.com/rihat99/mjlab.git mjlab-supermimic
cd mjlab-supermimic
uv sync --extra cu130            # Python 3.13, torch 2.14 for CUDA 13 (Blackwell)
PY=$PWD/.venv/bin/python
DATA=${MJLAB_DATA:-$PWD/../retarget_test/mjlab}
$PY -m pytest tests/test_supermimic_rl.py tests/test_supermimic_interface.py \
    tests/test_motion_library.py tests/test_tracking_kernels.py tests/test_tracking_kinematics.py
```

`test_supermimic_interface.py` reads clips from `$MJLAB_DATA/motions_lafan/` and skips
without them.

`--extra cu126` instead for NVIDIA drivers older than 580 (no Blackwell kernels).
The tools set `CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID`, `MUJOCO_GL=egl` and
`WANDB_MODE=disabled` from their `--gpu` argument.

## Data

All data lives under one root, `$MJLAB_DATA`, by default
`<repo root>/../retarget_test/mjlab` (`tools/paths.py`). The tools use:

- `motions_lafan/<clip>.npz`: the 40 LAFAN1 G1 clips at 50 Hz (441,140 frames,
  2.45 h), with the conversion logs and a README.
- `motions_seed/<filename>.npz`: all 142,220 BONES-SEED G1 clips at 50 Hz
  (51,915,358 frames, 93.3 GB), `manifest.json` (which clips the three filter steps
  keep: 129,043) and the conversion / filter logs.
- `runs/lafan_supermimic/<timestamp>[_<tag>]/`: one training run each.
  `params/env.yaml`, `params/agent.yaml` (full configs; `env.yaml` records the
  library's paths), `git/` (diff of the code it ran), `metrics.csv` (one row per
  logged iteration), `curves.png`, TensorBoard `events.out.tfevents.*`,
  `model_<it>.pt`, `eval_library_model_<it>.json` (library evaluation of that
  checkpoint), `figures/` (`plot_supermimic_run.py`), `videos/play/` (play).
  `runs/lafan_supermimic/library_<hash>.txt` records the sha256 of every clip of a
  training library (`sha256sum -c` re-checks it). The record lists absolute clip paths,
  so records written before the data moved name paths that no longer exist; the 40
  clips are unchanged (hashes match by file name) and a new record is written for the
  new paths.
- `viser/supermimic/<clip>/`: rendered videos, stills and `rollout.npz`.

Clip npz format (read by `MotionLoader` in `mdp/commands.py`; T frames, 30 bodies of
the G1, world frame). `MotionLoader` keeps only the 14 tracked bodies and stores
everything but the body positions in float16 on the GPU (572 B per frame; the SEED
kept set takes 25.1 GiB and loads in about 6.6 min):

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

Build `motions_seed/` from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
(`g1.tar.gz` extracted to `<seed>/g1/csv/<date>/<filename>.csv`, 120 fps, root in cm and
Euler degrees, joints in degrees), then filter it:

```bash
$PY tools/convert_seed.py <seed>/g1/csv --workers 96   # ~1 min on 96 cores
$PY tools/filter_seed.py <seed> --workers 96           # ~1 min, writes manifest.json
```

`convert_seed.py` writes the same npz as `csv_to_npz` (same resampling, velocities and
body states) in float64 numpy and CPU MuJoCo kinematics instead of a simulation per
clip. On 6 test clips its resampling matches `csv_to_npz`'s run in float64 to 1e-14,
and its body states match `csv_to_npz`'s to 2e-6 m from the same resampled states; end
to end the two differ by up to ~1e-3 because `csv_to_npz` interpolates in float32 and
its slerp holds the earlier frame when the root turns less than ~0.1° per 120 Hz frame. `manifest.json` gives, per clip,
`kept`, the `reasons` it was removed and its length; the name filter is SONIC's 40
keywords plus "50cm" and "look_over_fence", the floating-feet gate is
SuperMimic-Reproduced's (the paper gives none). A third step, ours and not the
paper's, drops clips in which a joint moves more than 1 rad between two 50 Hz frames
(retargeting glitches; the same rule as the colleague's SuperMimic-Reproduced pack).

| step | clips | removed |
|---|---|---|
| raw | 142,220 | |
| 1. name filter | 129,691 | 12,529 (paper 12,553) |
| 2. feasibility filter | 129,581 | 110 (paper 153; paper total 129,514) |
| 3. joint step > 1 rad | 129,043 | 538 (568 flagged, 30 already removed) |

Kept: 129,043 clips, 47,182,346 frames, 262.1 h. `train_library.py` trains on the kept
clips of `manifest.json` when the motions dir has one.

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

Analytic termination is the default; the learned-head arm adds
`--agent.policy.analytic-termination False` after `--`. A run of the
`-SelfCollision` task (see Configuration pointers) passes
`--task Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-SelfCollision` to `train_library.py`
and `TASK=Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-SelfCollision` to `eval_watch.sh`,
so that its evaluation uses the same physics. The two retrain arms of `task/retrain`
both use analytic termination and differ only in the contact physics: `uni_env` (the
default task) and `pelvis_analytic` (the `-SelfCollision` task). Anything after `--`
goes to mjlab's `train` unchanged; `--clips a,b` trains on a subset; `--print-cmd`
only prints the command.

Evaluate a checkpoint (default: the newest run and its last checkpoint) on its
training library; writes `eval_library_model_<it>.json` next to it (compact JSON:
aggregates and one record per rollout; `--num-envs` rollouts at a time):

```bash
$PY tools/train_library.py --eval --gpu 0 [--run-dir RUN | --checkpoint RUN/model_14000.pt]
```

Each clip record holds the fired termination (`fired`: `anchor_pos` = height test,
`anchor_ori` = tilt test, at frame `steps`) and the return per reward term. Diagnostic
flags go after `--`, with `--output-json RUN/eval_library_model_<it>_<variant>.json`
so the default JSON is kept: `--anchor-body torso_link` (needed for the 14k runs
below, trained with that anchor), `--fall-body pelvis` (fall test on another body),
`--self-collision False`, `--rollouts-per-clip 8` (per-clip failure fraction in
`clip_failure_fraction`), `--train-noise True` (training observation and reset
noise). A single rollout is not reproducible: the GPU simulation is not bitwise
deterministic and the policy amplifies the difference, so the same checkpoint can
fail 4 or 7 clips; count failures over several rollouts per clip.

BONES-SEED: train on the 129,043 kept clips (`--clips a,b` for a subset) and evaluate
on all of them. The library goes to mjlab as `<log-root>/library_<hash>_files.txt`;
each launch hashes the 85 GB of clips (~1 min) and loads the library (~6.6–10 min); training takes 0.26 s per iteration at 4096 envs and 42 GB of GPU memory, with `--agent.policy.seed-end-margin 10` (see `seed_end_margin` below; without it every SEED run collapsed). A
full evaluation pass at 4096 envs is 32 waves, 19,256 steps (~6 min of simulation at
17.6 ms per step, plus the load):

```bash
SEED="--motions-dir $DATA/motions_seed --log-root $DATA/runs/seed_supermimic"
nohup $PY tools/train_library.py --gpu 0 --iters 320000 --tag seed $SEED \
    > $DATA/runs/seed_supermimic/train.log 2>&1 &
$PY tools/train_library.py --eval --gpu 0 $SEED [--run-dir RUN]
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
survived steps ("survived"). These runs used BeyondMimic's ``torso_link`` as the
anchor body for termination, anchor rewards and relative body targets, and the
contact physics of the `-SelfCollision` task; the config now uses the paper's root
(``pelvis``) and the Uni-SuperMimic contact physics, so later runs are not directly
comparable.

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
shows the same level, not a better policy. Speed: these runs took 0.40 s per
iteration on an idle RTX PRO 6000 (paper: 0.11 s on an H100). The current code takes
0.24 s per steady-state iteration at 4096 envs on the same card (0.25 s in a training
run's log; 0.32 s with `--agent.precision.autocast none`): the objective's reward and
termination are compiled, the motion library is static for the CUDA graphs, host
syncs are gone from the world-model steps and the env step, and the world-model fit
and the policy update run in fp16 (see `PrecisionCfg` below; its effect on the
policy's quality at 14k is not yet measured).

Runs: `$DATA/runs/lafan_supermimic/2026-10-23_00-33-44_final_learnedhead/` and
`2026-10-23_00-33-45_final_analyticterm/`. Detailed report (curves, per-clip table,
the compile bug, speed, comparison with the colleague's run, local task notes outside this repository):
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-22_14-35_supermimic/REPORT.md`.

Retrain on `task/retrain`: two 14,000-iteration runs (seed 42, 4096 envs), both with
the pelvis anchor and fall test and analytic termination, differing only in the
contact physics: `uni_env` (the default task: no self-collision, friction
randomisation on all robot geoms, ground 0.3 in training) and `pelvis_analytic` (the
`-SelfCollision` task: the physics of the 14k runs above). Each checkpoint is
evaluated in its own task's play cfg with 8 rollouts per clip
(`--rollouts-per-clip 8`, no added noise), because a single rollout is one random draw
(see the evaluation note in Commands). Clip and survived are means over the 8 passes
of 40 clips; failed is the mean number of failed clips per pass (min–max over passes);
"1 rollout" is the checkpoint's `eval_watch` evaluation.

| run | iteration | eval clip | eval survived | failed per pass | failed, 1 rollout |
|---|---|---|---|---|---|
| uni_env | 10000 | 0.898 | 0.898 | 0.0 (0–0) | 0 / 40 |
| uni_env | 12000 | 0.910 | 0.910 | 0.1 (0–1) | 1 / 40 |
| uni_env | 14000 | 0.915 | 0.915 | 0.0 (0–0) | 0 / 40 |
| pelvis_analytic | 10000 | 0.869 | 0.888 | 2.4 (0–4) | 2 / 40 |
| pelvis_analytic | 12000 | 0.891 | 0.900 | 1.8 (0–3) | 0 / 40 |
| pelvis_analytic | 14000 | 0.884 | 0.899 | 1.2 (1–3) | 3 / 40 |

The same 8-rollout evaluation of the 14k torso runs above, with their physics and the
torso fall test: learned head 0.886 / 0.906 / 3.4 (2–5), analytic 0.884 / 0.898 / 1.5
(0–4); with the pelvis fall test the analytic run fails 5.1 per pass, mostly floor
rolls where the reference pelvis turns over (dance1_s1, dance2_s4, fall1_s1,
jumps1_s2), which `pelvis_analytic` does not fail at 14k. The 0 failures of `uni_env` need
its contact model at evaluation: with self-collision on (the `-SelfCollision` task)
the same checkpoints fail 28.8 / 24.9 / 16.2 clips per pass at 10k / 12k / 14k
(survived 0.89–0.91 before the fall), so they rely on the absence of self-contacts,
which a real robot has. `pelvis_analytic` evaluated with `--self-collision False`
fails 0.1 / 0.8 / 2.5 per pass. Both runs take 0.40 s per iteration (median over
12k–14k). Runs: `$DATA/runs/lafan_supermimic/2026-10-23_21-20-15_uni_env/` and
`2026-10-23_21-46-09_pelvis_analytic/`, evaluations in
`eval_library_model_<it>_copies8[_selfcoll|_noselfcoll].json`.

## Configuration pointers

- `SuperMimicRunnerCfg` (`src/mjlab/rl/supermimic/config.py`): every paper
  hyperparameter (Tables A.1–A.5); fields marked "(choice)" are not given in the
  paper. Override any field from the command line after `--`, e.g.
  `--agent.policy.horizon 16`.
- Contact physics (`unitree_g1_flat_tracking_supermimic_env_cfg` in
  `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`), as in Uni-SuperMimic and the
  colleague's implementation (the paper does not say): `self_collision=False` (no
  robot self-contacts) and `friction_all_geoms=True` (the startup friction U(0.3, 1.2),
  one value per env, on all 33 robot collision geoms instead of the 14 foot geoms, and
  the ground at 0.3 in training, because MuJoCo takes the larger friction of two
  equal-priority geoms and only the feet have priority 1). The play cfg sets every
  robot geom and the ground to 1.0. The task
  `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-SelfCollision` keeps the previous
  physics (`self_collision=True, friction_all_geoms=False`).
- Goal layout (`goal` of `unitree_g1_flat_tracking_supermimic_env_cfg`). The default
  `"reference"` is Uni-SuperMimic's deployable layout: the goal holds the 10 reference
  frames k_t .. k_t + 9 only, each as root position delta (3), root rotation (6D),
  root linear and angular velocity in the frame's own root frame (3 + 3), joint
  positions (29) and 0.1 x joint velocities (29), position and rotation expressed in
  the heading (yaw) frame of the current reference frame k_t - 1 (730 in all); the
  actor frame gains the 6D orientation error of the robot root to that current
  reference frame (noise U(-0.05, 0.05); 99 per frame, 990 with the history). No
  robot position enters the policy input, so deployment needs no global position
  estimate; per-step returns are expected a few hundredths below the previous layout,
  which showed the policy its full anchor position error. k_t - 1 is Uni's frame
  `f` (its reward scores `f + 1`); the goal frames are Uni's `f + 1 .. f + 10`.
  `"anchor"` is the previous layout (the reference anchor poses seen from the robot
  anchor, 670; actor 930): tasks `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-AnchorGoal`
  and `...-SelfCollision-AnchorGoal`, which every checkpoint from before 2026-10-24
  needs (`evaluate_library`, `tools/check_policy_grad.py --task`,
  `tools/check_reward_grad.py --task`).
- `PolicyCfg.analytic_termination` (default on): imagination uses the task's analytic
  soft termination instead of the learned head (the head keeps training, for
  logging). The learned head is kept for comparison only.
- `PrecisionCfg.compile` (default on): `torch.compile` plus CUDA graphs for the
  world-model loss, the imagined step, the state heads and the critic loss. The
  policy objective always runs eagerly: compiled as a whole, its forward matched
  eager but its backward did not (gradient cosine 0.52–0.54 to the eager gradient),
  and every run trained with it plateaued at a normalised return of 0.40. Its reward
  and termination are compiled each on its own, which keeps the eager gradient.
  `--agent.precision.compile False` runs everything eagerly.
  `tools/check_policy_grad.py` (4-6 minutes on one GPU) compares the compiled
  learner's policy and world-model gradients with the eager learner's on a
  checkpoint and fails if the whole-objective compile would no longer be caught
  (mean cosine 0.70–0.77 for that build). Its cosines are noisy: the 32-step imagined
  rollouts amplify TF32 rounding, so over all runs so far the 6-trial means of the
  TF32 build range 0.981–0.9999 (single trials down to 0.90 on the uni_env
  checkpoint) and of the fp16 build 0.914–0.9996 (single trials down to 0.50); the
  0.995 / 0.99 bars then fail on a correct build. With `--no-tf32` (fp32 matmuls
  everywhere) the compiled build matches eager at 0.999999 (every trial 0.999996 or
  more). `tools/check_reward_grad.py` (about 5 minutes) is the tight check: compiled
  vs eager reward, reward terms and termination on identical inputs, values and input
  gradients within 2e-6 relative (the learned head under fp16: 2e-3).
- `PrecisionCfg.autocast` (default `"bf16"`): the world-model loss, the imagined steps
  and the policy objective run under bfloat16 autocast. bf16 has fp32's exponent
  range, so there is no loss scaling: the gradient scalers are disabled and the
  `Metrics/*_grad_scale` columns are absent, as with `"none"`; its 8-bit mantissa
  rounds every Linear output and activation more coarsely than fp16 (11 bits).
  `"fp16"` runs the same under float16 autocast, each backward pass with its own
  dynamic gradient scaler (scales logged as `Metrics/*_grad_scale`, kept at 2^10 or
  more; an overflow skips the step and counts it in `Metrics/*_skipped_steps`; a log
  row with a scale below 2^12 or more than 0.5 % skipped steps prints a warning).
  Collection and the critic stay fp32. Unlike TF32, which rounds only the matmul
  inputs, fp16 rounds every Linear output and activation to fp16. fp16 policy-gradient
  cosine to the eager gradient (see `tools/check_policy_grad.py` above for the noise):
  6-trial means 0.9937–0.9996 on the 14k learned-head checkpoint (TF32 compiled
  0.99736–0.9999), 0.914–0.9993 on the uni_env checkpoint, 0.980 against an fp32
  (no TF32) reference. Over 1500 iterations the curves overlap the fp32 run's (sim
  return 0.511 vs 0.512 at 1500); the world-model scaler skipped 4 of 15,000 steps as
  its overflow threshold fell during training (harmless). The effect on quality over
  a full 14k run is not yet measured; `"none"` restores fp32 with TF32 matmuls.
- `PolicyCfg.seed_end_margin` (default 0): an imagined rollout is seeded only where
  `frame + horizon + seed_end_margin < clip_end`, so its last state, which bootstraps
  the critic targets and is the policy's terminal value, stays that many frames away
  from the clip's last frame. The critic is fit on the first `horizon` states only,
  so without a margin it is never fit on the last frames while still being evaluated
  there; on libraries with many clips that end mid-motion (SEED, LAFAN cut into 5-s
  chunks) that drove the critic loss to blow up after 1k–15k iterations. The SEED runs
  use `--agent.policy.seed-end-margin 10` (training README section above); the last
  `horizon + margin` frames of a clip are then never imagined.

## Known limitations and open items

- The exact operation `torch.compile` mis-differentiates in the policy objective is
  not localised (the synthetic-task test in `tests/test_supermimic_rl.py` does not
  exercise it; `tools/check_policy_grad.py` on the real task does).
- Ablation suggested by the gradient diagnostic, not run: the critic-bootstrap term
  alone collapsed the policy, reward-only improved more than the full objective over
  25 updates.
- Untested design decisions (only confounded runs): raw-metre goal scaling, critic
  error features. The anchor moved from torso to pelvis after the 14k runs; its
  effect is not measured yet.
- Training episodes terminate about ten times more often than in the colleague's
  run; evaluation failures are single rollouts per clip and change between
  checkpoints (2–7). Align the fall criterion and average a few rollouts per clip
  before the next comparison.
- Paper budget not run: 320k iterations, about 22 h at 0.25 s.
- BONES-SEED: converted, filtered and smoke-tested (40 iterations); no long run yet.
  Imagined rollouts start only where all H = 32 frames stay in the clip, so the last
  32 frames of every clip never seed one: 8.8 % of SEED frames (0.3 % of LAFAN), and
  the 8 clips shorter than 33 frames none at all.
- `render_viser_supermimic.py`: the second video rendered in the same process is
  3–10x slower than the first; render one ghost per process (`--ghosts reference`,
  then `--ghosts targets`) when time matters.
- `tests/test_tracking_kernels.py::test_recorded_rollout_parity` and
  `tests/test_tracking_kinematics.py::test_fk_matches_recorded_rollout` need
  rollouts recorded before the kernel refactor by a `record_baseline.py` that is not
  in this branch; they skip without `TRACKING_PARITY_BEFORE` /
  `TRACKING_PARITY_AFTER`.
