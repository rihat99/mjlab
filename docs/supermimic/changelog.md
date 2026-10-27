# Changelog of the SuperMimic fork

Changes of this fork relative to upstream mjlab at `45b1e263e`. Upstream's own changes are in
`docs/source/changelog.rst`.

## Added

- SuperMimic, model-based motion tracking over a clip library: the `mjlab.rl.supermimic` learner
  (world model fitted on 8-step windows of a replay buffer, deterministic policy trained by
  backpropagation through 32-step imagined rollouts under bfloat16 autocast, critic, return-based reset
  sampling) and the `Mjlab-Tracking-Flat-Unitree-G1-SuperMimic` task. The task side adds a multi-clip
  `MotionCommand` (`motion_file` takes a tuple of npz or a `.txt` file listing them; only the tracked
  bodies are kept, in float16 except the body positions, so that a 47 M-frame library fits on one GPU),
  the `motion_end` time-out, the paper's 93-dim proprioception (Table A.1) as the actor observation, a
  790-dim goal (Uni-SuperMimic's 10-frame reference, 730, then SONIC's orientation error of the robot
  anchor to the reference anchor of each of those frames, 60), pure tracking kernels with a
  differentiable forward kinematics, the `TrackingModelInterface` (`tasks/tracking/rl/model_interface.py`)
  that evaluates observation, goal, reward and termination on imagined states, the reset sampler
  (`tasks/tracking/mdp/return_sampler.py`) and a terminal-state recorder (`mdp/recorders.py`). `train`
  and `play` accept a motion library. The learner's hot paths run through `torch.compile` (with
  inductor's combo kernels) and CUDA graphs; imagination ends with the task's analytic termination
  (temperatures in `MotionCommandCfg.termination_z_temperature` and `termination_tilt_temperature`), and
  `PolicyCfg.seed_end_margin` (default 10) keeps the last imagined state away from the clip end, where the
  critic is never fit. Critic values are clamped to `max_step_reward / (1 - gamma)` in the λ-return
  targets and bootstraps (the runner refuses a reset sampler whose `g_max` differs), the critic sees the
  reference root pose in the robot root frame, imagined actions are noise-free, and the reset sampler
  caps a bin's softmax share at `MotionCommandCfg.softmax_cap` (50) times the uniform share. The task
  simulates full robot self-collision without a penalty, randomises the friction of every robot
  collision geom (ground at 0.3 in training) and the encoder bias per episode and the torso CoM once per
  env; the sim context is stored with every replay transition. The runner logs `Sim/constraint_overflow`
  and `Sim/contact_overflow`, the number of envs whose simulation dropped constraint rows or contacts
  beyond its budget since the last log row. `SuperMimicRunner.load` loads a checkpoint without optimiser
  states (a shipped policy) with fresh optimisers.
- The `evaluate_library` script for tracking tasks: each clip of a library once from frame 0 with a
  deterministic policy, in waves of at most `--num-envs` clips, into `eval_library_<checkpoint stem>.json`
  next to the checkpoint. One process evaluates several checkpoints on one env: a tuple of
  `--checkpoint-file` paths, or `--run-dir` for every `model_<it>.pt` of a run that has no results JSON
  yet (`--overwrite True`: all of them); `--watch True` also evaluates each checkpoint the training
  writes, until its process (`--pid`) has exited. The JSON records the checkpoint's iteration, file time
  and the run's training wall-clock, the git commit of the evaluating code with a dirty flag, and the
  robot's collision setting.
- Local logging without WandB (`mjlab.rl.local_log_writer`: TensorBoard events, `metrics.csv` with the
  training wall-clock per logged iteration and `curves.png` in the run dir; `--logger local` is the
  default); the WandB options, uploads and dependency are removed.
- `csv_to_npz` writes a local npz and takes `--atol` / `--rtol` for its velocity self-check.
- `tools/`: data conversion and filtering (LAFAN1, BONES-SEED), the training and evaluation launcher,
  evaluation reports, and video rendering of a policy and of its world model (see `usage.md`).
- `MjlabViserScene.ghost_cast_shadow`: ghost meshes can cast shadows.

## Changed

- The tracking tasks allocate 350 constraint rows per world (`njmax`, was 250): hard BONES-SEED clips
  with full robot self-collision reached 275, and MuJoCo Warp drops the rows beyond the budget.
