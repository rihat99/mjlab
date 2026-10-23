# mjlab, `vic` branch: variable impedance control for G1 motion tracking

This branch of [mjlab](https://github.com/mujocolab/mjlab) (fork
[rihat99/mjlab](https://github.com/rihat99/mjlab); upstream's README is on `main`) lets
the motion-tracking policy predict the PD gains of the Unitree G1's joint actuators
next to the joint targets, following the variable impedance idea of Bogdanovic, Khadiv
and Righetti, "Learning Variable Impedance Control for Contact Sensitive Tasks" (RA-L
2020, [arXiv:1907.07500](https://arxiv.org/abs/1907.07500)). It sits on top of mjlab's
BeyondMimic-style tracker ([arXiv:2508.08241](https://arxiv.org/abs/2508.08241)) and
was used to train trackers on G1 motions retargeted by BetterRetarget. Training and
logging are fully local (TensorBoard events, `metrics.csv`, `curves.png`; no WandB).

## Variable impedance as implemented

The G1's nominal gains follow BeyondMimic: per joint `kp0 = I * wn^2`,
`kd0 = 2 * zeta * I * wn` with `wn = 2 pi * 10 Hz`, `zeta = 2` and `I` the reflected
rotor inertia. The gain term (`JointGainAction`) adds one raw action `u` per joint,
squashed to a bounded log multiplier of the natural frequency:

- `freq` mode (task `-Vic-Freq`, 29 + 29 = 58 action dims):
  `log m = ln 2 * tanh(u / 2.5)`, `kp = m^2 kp0`, `kd = m kd0` (damping ratio fixed), so
  `m` lies in [0.5, 2] (natural frequency 5 to 20 Hz) and `kp` in [0.25, 4] x `kp0`;
- `freq_damp` mode (task `-Vic-FreqDamp`, 87 action dims): a second raw action per joint
  sets the damping-ratio multiplier `log r = ln 2 * tanh(u_r / 2.5)`, `r` in [0.5, 2],
  `kd = r m kd0`.

`u = 0` reproduces the nominal gains exactly. The gains are written every policy step,
per environment, into MuJoCo's built-in position actuators (`actuator_gainprm`,
`actuator_biasprm`, expanded per world in mujoco_warp); there is no Python-side PD. The
gain head explores with a fixed std of 0.5 (the joint head starts at 1.0 and stays
learnable): with a learnable gain-head std the entropy bonus grew it without bound once
the tanh saturated, and the gains went bang-bang.

The VIC tasks also change the rewards (`config/g1/env_cfgs.py`), the same for the
`-NewRew` control task, which keeps the fixed gains:

| term | formula | weight | tasks |
|---|---|---|---|
| `target_tracking_l2` | sum of (q_target - q)^2 after the step | -0.1 | NewRew, Vic-Freq, Vic-FreqDamp |
| `torque_rate_l2` | sum of (delta tau)^2 between policy steps | -1e-4 | NewRew, Vic-Freq, Vic-FreqDamp |
| `action_rate_l2` | joint-target head only | -0.1 | NewRew, Vic-Freq, Vic-FreqDamp |
| `gain_rate_l2` | sum of (delta log m)^2 (+ (delta log r)^2) | -0.1 | Vic-Freq, Vic-FreqDamp |
| `stiffness_cost` | sum of max(log m, 0) | -1e-2 | Vic-Freq, Vic-FreqDamp |

## What the branch adds to mjlab

- `src/mjlab/envs/mdp/actions/joint_gain.py`: `JointGainAction` / `JointGainActionCfg`
  (also `freeze()` to pin the gains, used by the evaluation).
- `src/mjlab/envs/mdp/rewards.py`: `action_rate_l2(..., action_name=...)` restricted to
  one action term, `action_term_slice`.
- `src/mjlab/tasks/tracking/config/g1/`: tasks
  `Mjlab-Tracking-Flat-Unitree-G1-Vic-Freq`, `-Vic-FreqDamp`, `-NewRew`
  (`__init__.py`, `env_cfgs.py`, `rl_cfg.py`).
- `src/mjlab/tasks/tracking/mdp/rewards.py`, `metrics.py`: the VIC reward terms and the
  `Episode_Metrics/*` terms (target-tracking error, torque saturation, gain statistics).
- `src/mjlab/tasks/tracking/rl/runner.py`: `TrackingRunnerCfg` (gain-head initial std,
  fixed gain-head std) and `Gains/*` training logs.
- `src/mjlab/tasks/tracking/scripts/evaluate.py`: smoothness metrics (`joint_acc_rms`,
  `torque_rate_rms`, `action_rate_joint`), gain metrics and per-step gains
  (`eval_<ckpt>_gains.npz`), `--freeze-gains True` (second pass with every joint's gain
  pinned to its time mean).
- `src/mjlab/rl/local_log_writer.py` and the training scripts (shared base of the fork
  branches): local TensorBoard + `metrics.csv` + `curves.png` logging, no WandB,
  logs finalised on a crash or SIGTERM.
- `tools/`: scripts to convert motions, train, queue, evaluate, play, render and plot
  (below); `tools/paths.py` holds the data root.

## Install

```bash
git clone -b vic https://github.com/rihat99/mjlab.git mjlab-vic && cd mjlab-vic
uv sync --extra cu130          # python 3.13, torch cu130; or --extra cu126 / cpu
PY=$PWD/.venv/bin/python       # the tools run with this interpreter
```

## Data

Everything lives under the data root `$MJLAB_DATA`, by default
`<repo root>/../retarget_test/mjlab` (`tools/paths.py`). Folders the tools use:

- `motions/<clip>.npz`: reference motions in mjlab's tracking format, written by
  `tools/convert_motion.py`: `fps` (1,) = 50, `joint_pos` / `joint_vel` (T, 29) in
  mjlab's G1 joint order, `body_pos_w` (T, 30, 3), `body_quat_w` (T, 30, 4) wxyz,
  `body_lin_vel_w` / `body_ang_vel_w` (T, 30, 3), robot bodies in mjlab order (body 0 =
  pelvis), float32.
- `retarget/betterretarget/<clip>.{npz,json}`: the retargeting results behind them
  (input of `convert_motion.py`; the input contract is in its docstring:
  BetterRetarget result format version 2, `time`, `fps`, `root_pos`, `root_quat` xyzw,
  `dof_pos`, `dof_names`, `dof_mask`, plus a json with `status` ok).
- `runs/<clip>/<run>/`: training runs (`train_tracker.py`, `train_queue.py`):
  `params/{env,agent}.yaml`, `git/mjlab.diff` (commit, status and diff of the code at
  launch), `metrics.csv`, `curves.png`, TensorBoard events, `model_<it>.pt` (the full
  runs keep every 5000 and the last, 29999), `<run>.onnx`,
  `eval_model_<it>[_<variant>].json` and, for gain tasks,
  `eval_model_<it>[_<variant>]_gains.npz`. `runs/<clip>/motion_<sha256[:12]>.npz` is the
  read-only motion copy the clip's runs trained on. Run names are
  `<date>_full[_<tag>][_fixstd]` for the 30k-iteration runs; tags: none = original
  tracker (task `Mjlab-Tracking-Flat-Unitree-G1`), `base` = the same task trained in the
  VIC queue, `newrew`, `vicfreq`, `vicfreqdamp`, `vicfreq_stiff1p5e-3` (stiffness cost
  1.5e-3); `_fixstd` = trained with the fixed gain-head std (now the default).
- `queue/`: `train_queue.py` status tables (`*.tsv`) and trainer stdout logs.
- `videos/<clip>/<tag>/`: `realtime.mp4`, `slow025.mp4` (0.25x). Viser renders
  (`render_viser.py`, tag `original` / `vicfreq` / `vicfreqdamp` / ...) add `t<T>.png`
  stills and `interactive/index.html` (open in a browser, orbit while it plays); the
  offscreen renders `native` / `native_randomized` (`render_force_ghost.py`, planned vs
  simulated contact forces) and `native_action_ghost` (`render_action_ghost.py`) add
  `sheet.png`.
- `rollouts/<clip>/<tag>.npz`: per-substep record of the rollout behind those videos
  (qpos, ctrl, torques, gains, feet heights; `native*.npz` also contact forces).
- `figures/<clip>/`: `gains_<tag>.png`, `gains_heatmap_<tag>.png`, `torques_<tag>.png`
  (`plot_gains_torques.py`).

Runs trained before the data root moved record the old absolute motion path in
`params/env.yaml`; the tools then use the file of the same name next to the run dir.

## Commands

All from the repo root, `PY=.venv/bin/python`; `$MJLAB_DATA` below is the data root
(export it, or write the default path). Every tool takes `--gpu N` (default 0) and sets
`CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID`, `MUJOCO_GL=egl` and
`WANDB_MODE=disabled` itself; `--help` prints the full docstring.

```bash
# Convert retargeted clips into motions/<clip>.npz (mjlab's csv_to_npz path, 50 Hz).
$PY tools/convert_motion.py --gpu 0 --input-dir $MJLAB_DATA/retarget/betterretarget

# Train one clip (30k iterations, 4096 envs as in the reported runs).
$PY tools/train_tracker.py --gpu 0 --clip amass_cmu_88_06 --iters 30000 \
    --tag full_vicfreq --task Mjlab-Tracking-Flat-Unitree-G1-Vic-Freq

# Many clips on several GPUs, K runs per GPU (optionally through CUDA MPS, --mps).
nohup $PY tools/train_queue.py --gpus 0 1 --slots 2 --iters 30000 --tag full_vicfreq \
    --task Mjlab-Tracking-Flat-Unitree-G1-Vic-Freq > queue.log 2>&1 &

# Evaluate (newest run of the clip, last checkpoint; task and motion from the run).
$PY tools/train_tracker.py --eval --gpu 0 --clip amass_cmu_88_06 [--run-dir DIR]
$PY tools/train_tracker.py --eval --gpu 0 --run-dir DIR -- --freeze-gains True

# Play headless: <run dir>/videos/play/rl-video-step-0.mp4, one motion pass.
$PY tools/train_tracker.py --play --gpu 0 --clip amass_cmu_88_06 [--run-dir DIR]

# Viser videos with the joint-target ghost, the interactive page and the rollout npz.
$PY tools/render_viser.py --gpu 0 --chrome CHROME --clip amass_cmu_88_06 \
    --run 2026-10-21_00-32-59_full_vicfreq_fixstd --stills 0.8 1.9 3.0
$PY tools/render_viser.py --gpu 0 --only-rollout --clip amass_cmu_88_06 --run RUN

# Gain and torque figures from the rollouts (needs rollouts/<clip>/<tag>.npz).
$PY tools/plot_gains_torques.py --clip amass_cmu_88_06 --tag vicfreq vicfreqdamp

# Offscreen ghost videos of the original tracker (newest run of the clip must be its
# plain _full run).
$PY tools/render_action_ghost.py --gpu 0 --clip amass_cmu_88_05
$PY tools/render_force_ghost.py --gpu 0 --clip amass_cmu_88_05 --br-dir DIR
```

`CHROME` is a Chromium headless shell
(`npx playwright install chromium-headless-shell`, installed under
`~/.cache/ms-playwright/chromium_headless_shell-*/`) or a wrapper that execs it with
`--no-sandbox`; on a machine without `libasound.so.2` the wrapper also puts an
extracted copy of the library (`apt-get download libasound2t64; dpkg-deb -x ...`) on
`LD_LIBRARY_PATH` (example in the `render_viser.py` docstring). GPU WebGL needs the
NVIDIA Vulkan ICD (`/usr/share/vulkan/icd.d/nvidia_icd.json`); `--software` renders
with SwiftShader instead. `render_force_ghost.py --br-dir` reads the planned contact
forces as `<clip>_br.npz` (keys in its docstring), which BetterRetarget's
`dump_br_plan.py` writes.

## Key results

Three clips retargeted by BetterRetarget (amass_cmu_88_06 aerial twist, amass_cmu_88_07
cartwheel, amass_accad_walk_turn_around walk with a 180 degree turn), one training seed
per arm, 4096 envs, 30k iterations, seed 42. Evaluation: mjlab `evaluate.py`, 1024 envs,
10 s, deterministic policy, observation noise on, pushes off. mpkpe = global,
r_mpkpe = root-relative; it90 = first iteration where the rolling-100 task reward
reaches 90 % of its final gain; sat = fraction of actuator-steps at the torque limit;
freeze = gains pinned to each joint's time mean; schedule replay = open-loop replay of
the per-frame mean gain schedule.

| clip | arm | success | mpkpe mm | r_mpkpe mm | ee_pos mm | it90 task | joint_acc rms | torque_rate rms | action_rate joint | sat % | mean kp/kp0 | log m std_time | log m at soft bound | freeze -> success | schedule replay -> success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 88_06 | orig | 0.986 | 131.7 | 47.4 | 62.6 | 3571 | 78.6 | 3.43 | 1.32 | 0.09 | 1 | – | – | – | – |
| 88_06 | NewRew | 0.993 | 143.2 | 51.5 | 69.0 | 3659 | 75.5 | 3.02 | 1.18 | 0.04 | 1 | – | – | – | – |
| 88_06 | Vic-Freq | 0.971 | 131.9 | 51.1 | 70.9 | 4334 | 86.1 | 3.43 | 0.76 | 0.18 | 0.66 | 0.26 | 0.08 | 0.000 | 0.914 |
| 88_06 | Vic-FreqDamp | 0.975 | 144.0 | 50.7 | 69.3 | 5815 | 85.8 | 3.44 | 0.81 | 0.25 | 0.70 | 0.28 | 0.08 | 0.000 | 0.746 |
| 88_06 | Vic-Freq stiff 1.5e-3 | 0.986 | 138.2 | 50.2 | – | 4570 | 87.7 | 3.45 | 0.71 | 0.22 | 0.70 | 0.28 | 0.11 | 0.000 | 0.873 |
| 88_07 | orig | 1.000 | 133.9 | 57.0 | – | 4328 | 68.7 | 2.80 | 1.46 | 0.72 | 1 | – | – | – | – |
| 88_07 | NewRew | 1.000 | 134.2 | 59.4 | – | 6745* | 67.9 | 2.48 | 1.39 | 0.35 | 1 | – | – | – | – |
| 88_07 | Vic-Freq | 0.998 | 130.1 | 61.7 | – | 6655* | 78.7 | 2.62 | 0.99 | 0.37 | 0.58 | 0.30 | 0.15 | 0.000 | 0.949 |
| 88_07 | Vic-FreqDamp | 1.000 | 138.8 | 57.4 | – | 6111* | 76.8 | 2.76 | 1.01 | 0.42 | 0.65 | 0.32 | 0.07 | 0.000 | 0.911 |
| walk | base | 1.000 | 80.9 | 32.4 | – | 1525 | 47.9 | 2.37 | 0.64 | 0.00 | 1 | – | – | – | – |
| walk | NewRew | 1.000 | 96.9 | 32.3 | – | 1584 | 46.1 | 2.09 | 0.57 | 0.00 | 1 | – | – | – | – |
| walk | Vic-Freq | 1.000 | 85.1 | 32.6 | – | 2032 | 51.4 | 2.32 | 0.44 | 0.05 | 0.47 | 0.18 | 0.26 | 0.000 | 0.106 |
| walk | Vic-FreqDamp | 0.999 | 88.6 | 32.7 | – | 2566 | 47.7 | 2.24 | 0.40 | 0.10 | 0.50 | 0.19 | 0.17 | 0.000 | 0.333 |

\* On 88_07 every arm stalls at an episode length of 230–330 before breaking out at
about 2k (orig), 3k (FreqDamp), 5k (Freq) and 6k (NewRew) iterations, so its convergence
column is uninformative at n = 1. Evaluation noise: success ±0.008, global mpkpe
±1.9 mm; run-to-run spread (two near-identical Vic-Freq runs): success 0.015, global
mpkpe 6.3 mm, r_mpkpe 0.9 mm.

In short: the gains work as a mechanism and the policy uses them heavily, but they do
not improve tracking, convergence or physical smoothness over the fixed-gain NewRew
control with the same rewards. Root-relative error equals the control's (2–4 mm worse
than the original tracker on the dynamic clips; the global-mpkpe differences are root
drift). Convergence is 0.3–2k iterations slower on 88_06 and the walk. The joint
targets move less, but joint acceleration rises 14–16 % on the dynamic clips. The
gains are not constant: each joint follows a strong motion-phase schedule (kp sweeping
0.25–3.9 x nominal within a stride), and every Vic policy falls within 2 s when its
gains are frozen.

Detailed report (method, calibration, protocol, all tables and figures, local task notes outside this repository):
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-20_20-26_vic_gains/REPORT.md`.
Videos and gain / torque figures of the 88_06 trackers:
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-21_13-00_vic_video/REPORT.md`;
the viser renderer and its look:
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-20_17-53_viser_sim_video/REPORT.md`;
the original trackers on BetterRetarget motions:
`/home/rikhat.akizhanov/better/tracking/BetterRetarget/agent_notes/tasks/2026-10-19_21-13_br_mjlab_tracking/REPORT.md`.

## Limitations and open items

- One training seed per arm: effects smaller than the run spread (about 6 mm global
  mpkpe, 0.015 success) are not established; 2–3 seeds per arm are needed.
- Untested: a lower-dimensional gain head (one multiplier per leg or body group, best
  in the only legged comparison of learned gains), harder settings (pushes, friction
  randomisation, a motion the original tracker fails such as amass_cmu_88_05), removing
  the entropy bonus from the gain dims instead of fixing their std.
- The adaptive learning rate runs 1.8–2.5 x lower on the Vic runs (the KL is summed over
  58 / 87 action dims): a confound for the convergence comparison.
- With the fixed gain-head std the mean gain keeps drifting toward the soft bound
  (0.08–0.26 of env-steps at the bound at 30k iterations; on the walk 8 of 29 joints sit
  at about 0.25–0.3 x kp0).
- The entropy-bonus explanation of the gain collapse with a learnable std is plausible
  but not isolated.
- Evaluation loops the motion by teleporting to frame 0 (only the first pass is
  clean); `torque_saturation` is measured at the policy step; the freeze test uses one
  per-joint vector for all envs.
- `render_action_ghost.py` and `render_force_ghost.py` build the original task
  (`Mjlab-Tracking-Flat-Unitree-G1`) and take the clip's newest run, which must end in
  `_full`: they render original-tracker runs only, and not on clips that have later VIC
  runs (88_06, 88_07, the walk); `render_viser.py` handles every task and takes `--run`.
