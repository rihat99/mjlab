"""SuperMimic runner configuration. Defaults are the paper's Table A.5 unless noted."""

from dataclasses import dataclass, field

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class WorldModelCfg:
  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  """Dynamics MLP hidden layers (Linear -> LayerNorm -> SiLU each)."""
  window_k: int = 8
  """Autoregressive window length of the k-step loss."""
  batches_per_update: int = 10
  """Adam steps per iteration, each on ``batch_size`` windows."""
  batch_size: int = 2048
  """Windows per Adam step."""
  lr: float = 2e-4
  """Adam learning rate (before the schedule)."""
  max_grad_norm: float = 1.0
  """Global gradient-norm clip."""
  nominal_floor: tuple[float, ...] = (3e-3, 5e-3, 2e-2, 5e-2, 3e-3, 5e-2)
  """(choice) Lower bound of the nominal increment scale per block: position (m),
  rotation (rad), linear velocity (m/s), angular velocity (rad/s), q (rad),
  qd (rad/s)."""
  statistics_horizon: int = 100
  """(choice) The input normaliser and the nominal increment scales are fitted at the
  end of the prefill, then updated before every world-model fit from the logged (real)
  transitions of its sampled windows, never from imagined states (frozen statistics
  fall behind the policy's actions): rsl_rl's running mean / variance until they hold
  this many fits' worth of samples, then an exponential moving average that weighs
  each fit's samples by 1 / (statistics_horizon + 1)."""


@dataclass
class PolicyCfg:
  encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  """Motion encoder MLP hidden layers."""
  latent_dim: int = 32
  """Size of the tanh-bounded motion latent z."""
  decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  """Decoder MLP hidden layers."""
  activation: str = "elu"
  """Hidden activation of encoder and decoder."""
  history_length: int = 10
  """Observation frames the policy sees; must match the env's actor history."""
  lr: float = 2e-4
  """Adam learning rate (before the schedule)."""
  betas: tuple[float, float] = (0.9, 0.95)
  """Adam betas."""
  max_grad_norm: float = 1.0
  """Global gradient-norm clip, applied first."""
  max_grad_value: float = 0.1
  """Element-wise gradient clip, applied after the norm clip."""
  num_rollouts: int = 4096
  """Imagined rollouts per policy update."""
  horizon: int = 32
  """Imagination horizon H."""
  seed_end_margin: int = 10
  """(choice) Seeds need ``frame + horizon + seed_end_margin < clip_end``: the
  bootstrap state s_H stays this many frames away from the clip's last frame. The
  critic is fit on s_0..s_{H-1} only; without a margin, V(s_H) at the ends of clips
  cut mid-motion drove the critic loss to blow up."""
  gamma: float = 0.99
  """Discount factor (imagination, critic targets and reset-sampler returns)."""
  normalizer_freeze_iteration: int = 20000
  """(choice) Iteration after which the observation/goal statistics stop updating."""


@dataclass
class CriticCfg:
  hidden: tuple[int, ...] = (256, 128)
  """Critic MLP hidden layers (ELU)."""
  lr: float = 2e-4
  """Adam learning rate (before the schedule)."""
  max_grad_norm: float = 1.0
  """Global gradient-norm clip."""
  epochs: int = 5
  """Passes over the imagined batch per iteration."""
  minibatches: int = 4
  """Shuffled minibatches per epoch."""
  lam: float = 0.95
  """Lambda of the lambda-return targets (Eq. 3)."""


@dataclass
class CollectionCfg:
  action_noise_std: float = 0.1
  """Std of the Gaussian noise added to the policy action in the simulator (it serves
  the world-model fit; imagined actions are noise-free, paper Alg. A.1)."""
  replay_steps: int = 1000
  """Replay buffer capacity in steps per env."""
  sim_steps_per_iteration: float = 2.5
  """Vectorised sim steps per iteration (update-to-data ratio 0.4); a fractional value
  alternates, e.g. 2, 3, 2, 3."""
  prefill_steps: int = 64
  """(choice) Sim steps collected before the first update; must exceed window_k."""
  warmup_world_model_iterations: int = 500
  """(choice) Iterations with world-model updates only, before policy and critic."""


@dataclass
class ScheduleCfg:
  lr_decay_iterations: int = 250_000
  """Iterations of the cosine decay; the rate stays at its final value afterwards."""
  lr_final_ratio: float = 0.03
  """Final learning rate as a fraction of the initial one (all three optimisers)."""


@dataclass
class PrecisionCfg:
  tf32: bool = True
  """Allow TF32 matmuls and convolutions (those that ``autocast`` leaves in fp32)."""
  autocast: bool = True
  """(choice) The world-model loss, the imagined steps and the policy objective run
  under bfloat16 autocast (matmuls in bf16 with fp32 accumulation, about twice the
  TF32 rate; every Linear output and activation is rounded to bf16's 8-bit mantissa).
  bf16 has fp32's exponent range, so no loss scaling. Collection and the critic fit
  stay fp32. False: fp32 throughout (TF32 matmuls with ``tf32``)."""


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  num_steps_per_env: int = 1
  """Unused by SuperMimic (sim steps come from ``collection``)."""
  max_iterations: int = 320_000
  save_interval: int = 4000
  log_interval: int = 50
  """Iterations between two log rows."""
  model_interface: str = ""
  """The task's ModelInterface class as ``"module.path:ClassName"``; it is built as
  ``cls(env.unwrapped)``."""
  world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)
  collection: CollectionCfg = field(default_factory=CollectionCfg)
  schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
  precision: PrecisionCfg = field(default_factory=PrecisionCfg)
