# Using the SuperMimic fork

Install, data, training, evaluation, videos and the configuration knobs. Every command runs from the repo
root with the project interpreter, `PY=$PWD/.venv/bin/python`. The GPU tools take `--gpu N` (an
`nvidia-smi` index; set `CUDA_DEVICE_ORDER=PCI_BUS_ID` if your CUDA order differs) and set
`CUDA_VISIBLE_DEVICES` themselves.

## Install

```bash
git clone -b supermimic https://github.com/rihat99/mjlab.git mjlab-supermimic
cd mjlab-supermimic
uv sync --extra cu130            # Python 3.13, torch for CUDA 13 (Blackwell)
PY=$PWD/.venv/bin/python
$PY -m pytest tests/
```

`--extra cu126` instead for NVIDIA drivers older than 580 (no Blackwell kernels).

## Data

Everything the tools read and write lives inside the repo, gitignored, at the paths of `tools/paths.py`;
`$MJLAB_DATA`, when set, replaces `data/`, and any folder may be a symlink to a larger disk.

| path | content |
|---|---|
| `data/motions/seed/` | BONES-SEED G1 clips (`<clip>.npz`, 50 Hz), `manifest.json` (which clips the filters keep), `groups.json` (Table A.8 motion groups); the default library |
| `data/raw/bones_seed/` | the BONES-SEED download (`g1/csv/`, `metadata/`) |
| `data/motions/lafan/` | the 40 LAFAN1 G1 clips |
| `runs/seed/`, `runs/lafan/` | one folder `<timestamp>[_<tag>]/` per training run, plus `library_<hash>.txt` (sha256 of every clip of a training library) and `library_<hash>_files.txt` (its path list) |
| `videos/<clip>/`, `videos/imagine/<clip>/` | outputs of `tools/view.py --video` and `tools/imagine_video.py` |

A run folder holds `params/env.yaml` and `params/agent.yaml` (the full configs, with the library's clip
list), `metrics.csv` (one row per logged iteration, with `training_wall_seconds`), `curves.png`,
TensorBoard events, `model_<it>.pt` (every 4000 iterations), `eval_library_model_<it>.json` and
`figures/`.

Clip npz (T frames, the 30 bodies of the G1, world frame): `fps` (1,) = 50; `joint_pos`, `joint_vel`
(T, 29); `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` (T, 30, 3); `body_quat_w` (T, 30, 4), wxyz.

**BONES-SEED** ([bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed), `g1.tar.gz`,
120 fps CSVs, extracted into `data/raw/bones_seed/`; its own licence terms). `convert_seed.py` resamples
to 50 Hz on the CPU; `filter_seed.py` applies the paper's name filter (SONIC's keywords plus "50cm" and
"look_over_fence", except `cartwheel`, `handstand`, `handstand_ff_`, `kozak_dance`, which we keep) and
feasibility filter and writes `manifest.json`: 142,220 clips, 130,063 after the name filter, 129,951 kept
(263.9 h; the paper keeps 129,514). `motion_groups.py` writes `groups.json`.

```bash
$PY tools/convert_seed.py data/raw/bones_seed/g1/csv --workers 96
$PY tools/filter_seed.py data/raw/bones_seed --workers 96
$PY tools/motion_groups.py
```

**LAFAN1** ([LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
`g1/*.csv`, 30 fps; CC BY-NC-ND 4.0), through mjlab's `csv_to_npz` on the GPU:

```bash
$PY tools/convert_lafan.py <LAFAN1_Retargeting_Dataset>/g1 --gpu 0 [--clips walk1_subject1,...]
```

## Train

`tools/train_library.py` trains on the kept clips of `--motions-dir` (default `data/motions/seed`) into
`--log-root` (default `runs/seed`), 4096 envs. It records the library (`library_<hash>.txt`; training
refuses to start if the recorded files changed), passes the clip list to mjlab's `train` and replaces
itself with it, so a `nohup`'ed PID is the trainer's. Anything after `--` goes to `train` unchanged
(e.g. `-- --agent.policy.horizon 16`); `--clips a,b` trains on a subset; `--print-cmd` prints the command
only.

```bash
mkdir -p runs/seed
nohup $PY tools/train_library.py --gpu 0 --iters 160000 --tag seed > runs/seed/train_seed.log 2>&1 &
TRAIN_PID=$!
# once the log names the run dir RUN: evaluate each checkpoint as it appears
nohup $PY tools/train_library.py --eval --watch --gpu 0 --run-dir RUN --pid $TRAIN_PID \
    > runs/seed/eval_watch_seed.log 2>&1 &
```

LAFAN1: `--motions-dir data/motions/lafan --log-root runs/lafan --iters 14000`. The watcher waits for the
run's `params/env.yaml`, loads the library once and evaluates every checkpoint until the trainer exits.

## Evaluate

The paper's protocol (App. F.1): every clip once from frame 0, play config (friction 1.0, no noise or
randomisation), deterministic policy, in waves of `--num-envs` clips. A clip fails when the pelvis height
error exceeds 0.25 m or its tilt error 0.8. The JSON next to the checkpoint holds the normalised return
per clip length and per survived step (mean over clips and frame-weighted), the failed clips, one record
per clip with the terminations that fired and the return per reward term, the run's training wall-clock at
the checkpoint and, under `provenance`, the evaluating commit (`dirty` flag) and the robot's collision
setting. MuJoCo Warp is not bitwise deterministic across processes: a repeat evaluation can move a few
clips between failed and not failed.

```bash
$PY tools/train_library.py --eval --gpu 0 [--run-dir RUN | --checkpoint RUN/model_160000.pt]
```

Without `--checkpoint` it evaluates every checkpoint of the run (default: the newest run) that has no JSON
yet (`-- --overwrite True`: all). The library is the one the run's `params/env.yaml` records; if that list
or a file is not on this machine the tool stops and asks for `--motions-dir`, whose kept clips then
replace it. A few clips, without touching the run's JSON:

```bash
$PY -m mjlab.tasks.tracking.scripts.evaluate_library Mjlab-Tracking-Flat-Unitree-G1-SuperMimic \
    --checkpoint-file RUN/model_14000.pt --output-json /tmp/eval.json \
    --motion-files "('data/motions/lafan/dance1_subject3.npz', 'data/motions/lafan/fallAndGetUp3_subject1.npz')"
```

Reports: `report_groups.py` writes the Table A.8 group table of the last JSON (`.md`, `.csv`, `.png`;
groups from `groups.json` next to the clips or `--groups FILE`) and, over several JSONs, the return and
failed clips vs iteration and vs training hours, a per-clip heatmap and a table per checkpoint;
`plot_supermimic_run.py` draws the training curves of `metrics.csv`. Both write to `RUN/figures/`.

```bash
$PY tools/report_groups.py RUN/eval_library_model_*.json [--groups FILE] [--out-dir DIR]
$PY tools/plot_supermimic_run.py RUN [--hours]
```

## Watch a policy

`tools/view.py` shows one clip live in mjlab's viser viewer (reference ghost, pause, frame scrubber, the
run's checkpoints to switch between); it prints its URL (port 8080 or the next free one; from another
machine `ssh -L 8080:localhost:8080 HOST` first). The clip comes from the run's recorded library, or from
`--motions-dir`. Anything after `--` goes to mjlab's `play` (`-- --device cpu`).

`--video` plays the clip once as the evaluation does (it prints the normalised return and whether the
policy fell) and renders 20 s of it (by default the 20 s with the most root motion, `--window-start`
to choose) as a 1920x1080, 50 fps `videos/<clip>/reference.mp4` with the reference as a blue ghost, or
`targets.mp4` with `--ghost targets` (orange ghost = joint targets), plus `rollout.npz`; `--stills T ...`
writes PNG stills instead. Frames come from a headless Chromium (Playwright's headless shell,
`npx playwright install chromium-headless-shell`, or `--chrome PATH`), which needs `libasound.so.2`; the
tool prints how to get it without root.

```bash
$PY tools/view.py --run-dir checkpoints/seed --clip side_roll_R_001__A304_M --gpu 0
$PY tools/view.py --checkpoint RUN/model_14000.pt --clip dance1_subject3 --video --gpu 0
```

`tools/imagine_video.py` puts the simulator and the world model side by side: the clip is played once
as above, then from 4 start steps (`--starts` to choose) the policy acts `--horizon` steps inside the world
model from the simulated state. It writes `videos/imagine/<clip>/imagine_h<H>.mp4` (simulation left,
imagination right in orange), `imagine_h<H>.npz` and `imagine_error_h<H>.png` (root and joint errors
over the horizon); `--no-video` skips the render.

```bash
$PY tools/imagine_video.py --run-dir checkpoints/seed --clip cartwheel_R_003__A135 \
    --horizon 32 --horizon 128 --gpu 0
```

## The shipped checkpoint folder

`checkpoints/seed/` holds `model_<it>.pt` (policy, critic, world model and reset-sampler state, without
optimiser states, to stay under GitHub's 100 MB file limit) and the run's `params/`. Its evaluation JSON
and `figures/` stay local (gitignored). It serves as `--run-dir` for `train_library.py --eval`,
`view.py` and `imagine_video.py`; its `params/env.yaml` records the clip list under the training
machine's paths, so on another machine add `--motions-dir data/motions/seed` (evaluation then uses the
kept clips of `manifest.json`, the same 129,951 clips if your filter run matches). Loading it whole
(`SuperMimicRunner.load` without a part selection) starts fresh optimisers.

## Configuration

- `SuperMimicRunnerCfg` (`src/mjlab/rl/supermimic/config.py`): the paper's hyperparameters (Table A.5);
  fields marked "(choice)" are not in the paper. `PolicyCfg.seed_end_margin` (10) keeps the last state
  of every imagined rollout that many frames before the clip end (the critic is never fit there; without
  it the critic loss blew up on BONES-SEED). `CollectionCfg.action_noise_std` (0.1) acts in the
  simulator only; imagined actions are noise-free. `PrecisionCfg.autocast` runs the world-model loss and
  imagination under bfloat16 (off: fp32 with TF32 matmuls).
- Critic values are clamped to `max_step_reward / (1 - gamma)` (= 10) in the λ-return targets, the
  policy's bootstrap and the reset sampler's bootstrap; the runner refuses a `MotionCommandCfg.g_max`
  that differs.
- Task (`unitree_g1_flat_tracking_supermimic_env_cfg`, `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`):
  pelvis anchor; full robot self-collision without a penalty; friction U(0.3, 1.2) on every robot
  collision geom with the ground at 0.3 in training (1.0 in the play config); encoder bias and friction
  resampled per episode, torso CoM offset once per env; actor = Table A.1's 93 values x 10 steps; goal =
  10 reference frames in the reference heading frame (730) plus the reference pelvis orientation seen
  from the robot pelvis for each of them (60).
- `MotionCommandCfg` (`src/mjlab/tasks/tracking/mdp/commands.py`): `softmax_cap` (50, a bin's softmax
  share capped at 50x uniform), `bin_return_alpha`, `past_weight_alpha`, and the temperatures of the
  analytic soft termination used in imagination (`termination_z_temperature`,
  `termination_tilt_temperature`).

## Compute

On one RTX PRO 6000 (Blackwell), 4096 envs: LAFAN1 about 0.21 s per iteration; BONES-SEED 0.22 s per
iteration and about 43 GB of GPU memory, 9.7 h of training for 160k iterations, after 35-70 min of
library loading (CPU-bound). A full-library watch evaluation needs another 29 GB and 10-12 min per
checkpoint while the trainer runs (4.4 min alone), and slows the trainer to about 0.36 s per iteration.

## Contact budget

MuJoCo Warp keeps a fixed number of constraint rows per world (`njmax`) and silently drops the rest. With
full self-collision, hard BONES-SEED clips reached 275 rows; the tracking tasks use 350. A CPU sweep of the
library with the training reset noise found rare floor-exercise reset poses at up to 381 rows, so a run
can still overflow for a few substeps after such a reset. The runner logs `Sim/constraint_overflow` and
`Sim/contact_overflow` (envs that dropped rows or contacts since the last log row; an overflow on an
episode's last step is missed), and MuJoCo Warp prints a "please increase" line in the training log.
