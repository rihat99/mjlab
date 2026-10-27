"""Simulator vs world-model imagination of a SuperMimic policy, side by side.

The clip is played once in MuJoCo as the library evaluation plays it (the protocol of
``render_viser.rollout``: play cfg, 1 env, deterministic policy, from frame 0), keeping
the floating-base state s_t (root pose and velocity, joint angles and velocities),
frame k_t, observation and action of every policy step. From each start step t0 the
policy then acts for ``--horizon`` steps inside the world model, as the policy update
imagines (``SuperMimic.imagine_step``, bf16 autocast as in training): the observation
history of t0 is rebuilt from the logged states, the policy acts on it and the
reference goal, the world model predicts the next state, whose observation joins the
history. Both sides are noise-free (the play cfg has no observation noise; training
imagination adds observation noise only, imagined actions are noise-free). The sim
context is the play cfg's (friction 1.0, no encoder bias, no CoM offset), normalised
on the training cfg's DR ranges. Imagined states hold the root pose and the joint
angles, so both robots are drawn directly from their states. Rewards of both sides
come from the task's reward head (fp32); the env's own rewards are kept too.

Writes ``videos/imagine/<clip>/``, per horizon H:

- ``imagine_h<H>.mp4``: the simulated robot and, OFFSET_M to its right on screen, the
  imagined one in orange; per start H + 1 frames at 50 / ``--slowdown`` fps, the
  first and last held;
- ``imagine_h<H>.npz``: sim and imagined states, actions, rewards and errors per start;
- ``imagine_error_h<H>.png``: root height and two joint angles over the horizon (sim
  solid, imagined dashed), root position, orientation and mean joint errors.

    PY=<repo>/.venv/bin/python
    $PY tools/imagine_video.py --run-dir checkpoints/seed --clip cartwheel_R_003__A135 \\
        --horizon 32 --horizon 128 --gpu 6 [--starts 0,90,180]

``--starts`` are policy steps of the rollout (default: 4 evenly spaced over the steps
that training accepts as seeds, frame + H + seed_end_margin < clip end). Chromium
needs ``libasound.so.2`` (see ``tools/view.py``).
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from paths import VIDEOS_DIR
from PIL import Image, ImageDraw, ImageFont
from render_viser import (
  BACKGROUND_RGB,
  CAMERA_FOV_DEG,
  CAPTION_PX,
  CAPTION_RGB,
  HEIGHT,
  WIDTH,
  HeadlessBrowser,
  build_scene,
  camera_path,
  check_chrome,
  find_chrome,
  free_port,
  ghost_model_of,
  view_direction,
)
from train_library import TASK, latest_checkpoint
from view import resolve_motion

IMAGINED_RGB, CAPTION_IMAGINED_RGB = (0.85, 0.3, 0.03), (215, 90, 0)
OFFSET_M = 1.0  # the imagined robot stands this far right of the simulated one
_RIGHT = np.cross(-view_direction(), (0.0, 0.0, 1.0))
SHIFT = OFFSET_M * _RIGHT / np.linalg.norm(_RIGHT)  # (3,) world, horizontal
CAMERA_DISTANCE_M = 3.6
LABEL_FORWARD_M = 0.45  # robot labels this far toward the camera from each root
_TOWARD_CAMERA = view_direction()[:2] / np.linalg.norm(view_direction()[:2])
HOLD_FIRST_S, HOLD_LAST_S = 0.5, 1.0  # video time
REPORT_STEPS = (8, 16, 32, 64, 128)
PLOT_JOINTS = ("left_knee_joint", "right_hip_pitch_joint")


def simulate(checkpoint: Path, motion: Path) -> tuple:
  """Runner (policy and world model loaded) and the rollout: state (T + 1, S),
  frame (T,), newest observation frame (T, O), action (T, A), env reward (T,)."""
  import mjlab.tasks  # noqa: F401 -- registers the tasks
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.scripts.evaluate_library import make_library_env
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  agent_cfg = load_rl_cfg(TASK)
  vec_env = make_library_env(TASK, [str(motion)], "cuda:0", agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None, f"{TASK} registers no runner"
  runner = runner_cls(vec_env, asdict(agent_cfg), device="cuda:0")
  parts = {"actor": True, "world_model": True}
  runner.load(str(checkpoint), load_cfg=parts, strict=True, map_location="cuda:0")
  policy, iface = runner.get_inference_policy(device="cuda:0"), runner.interface
  rows: dict[str, list] = {k: [] for k in ("state", "frame", "obs", "action", "reward")}
  obs, _ = vec_env.reset()
  while True:
    rows["state"].append(iface.state_from_sim()[0])
    rows["frame"].append(iface.frame_from_sim()[0])
    rows["obs"].append(runner.alg.policy.frames_from_env_layout(obs["actor"])[0, -1])
    with torch.no_grad():
      action = policy(obs)
    obs, reward, dones, _ = vec_env.step(action)
    rows["action"].append(action[0])
    rows["reward"].append(reward[0].clone())  # the env reuses its reward buffer
    if bool(dones[0]):
      if bool(vec_env.unwrapped.termination_manager.terminated[0]):
        raise SystemExit("the policy falls on this clip: choose a surviving one")
      rows["state"].append(iface.terminal_state_snapshot()[0])
      return runner, {key: torch.stack(value) for key, value in rows.items()}


def play_context(iface) -> torch.Tensor:
  """Normalised context (1, C) of the play cfg's sim on the training DR ranges.

  The play env has no DR terms, so the interface lacks the encoder-bias range its
  observation head needs to turn the context back into a bias: it is set here.
  """
  from mjlab.tasks.registry import load_env_cfg

  events = load_env_cfg(TASK).events
  low, high = events["foot_friction"].params["ranges"]
  bias_low, bias_high = events["encoder_bias"].params["bias_range"]
  com = np.array([events["base_com"].params["ranges"][axis] for axis in range(3)])
  iface._bias_range = (0.5 * (bias_low + bias_high), 0.5 * (bias_high - bias_low))
  friction = (1.0 - 0.5 * (low + high)) / (0.5 * (high - low))
  bias = -iface._bias_range[0] / iface._bias_range[1] * np.ones(iface.nq)
  offset = -com.mean(1) / (0.5 * (com[:, 1] - com[:, 0]))
  context = np.concatenate(([friction], bias, offset))
  return torch.tensor(context, dtype=torch.float32, device="cuda:0").unsqueeze(0)


@torch.no_grad()
def imagine(runner, sim: dict, starts: list[int], horizon: int, context) -> dict:
  """Noise-free imagined rollouts from the sim states at ``starts`` (B), as the policy
  update runs them: states (B, H + 1, S), actions (B, H, A), rewards and soft
  terminations (B, H); max |difference| (B,) of the first observation frame and the
  first action to the sim's."""
  alg, iface = runner.alg, runner.interface
  history, num = alg.policy_cfg.history_length, len(starts)
  t0 = torch.tensor(starts, device=context.device)
  # History rows before the episode start repeat its first row, with no prev action.
  rows = (t0[:, None] - torch.arange(history - 1, -1, -1, device=t0.device)).clamp(0)
  prev = sim["action"][(rows - 1).clamp(0)] * (rows > 0)[..., None]
  ctx = context.expand(num, -1)
  zeros = torch.zeros(num, history, iface.obs_dim, device=t0.device)
  frames = alg.history_frames(sim["state"][rows], prev, ctx, sim["frame"][rows], zeros)
  obs_error = (frames[:, -1] - sim["obs"][t0]).abs().amax(-1)
  s, frame, a_prev = sim["state"][t0], sim["frame"][t0], prev[:, -1]
  features, goal = alg.state_heads(s, frame)
  out: dict[str, list] = {"state": [s], "action": [], "reward": [], "termination": []}
  autocast = runner.cfg["precision"]["autocast"]
  for _ in range(horizon):
    with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
      action, s_next, frames, next_frame, features, goal = alg.imagine_step(
        frames, goal, s, features, ctx, frame, zeros[:, 0]
      )
    action = action.float()
    out["reward"].append(iface.reward(s_next, action, a_prev, frame))
    out["termination"].append(iface.termination_soft(s_next, frame))
    out["state"].append(s_next)
    out["action"].append(action)
    s, a_prev, frame = s_next, action, next_frame
  result = {key: torch.stack(value, 1).cpu().numpy() for key, value in out.items()}
  first = np.abs(result["action"][:, 0] - sim["action"][t0].cpu().numpy())
  return result | {"obs_error": obs_error.cpu().numpy(), "action_error": first.max(-1)}


def yaw(quat: np.ndarray) -> np.ndarray:
  """Heading (...,) rad of quaternions (..., 4) wxyz."""
  w, x, y, z = np.moveaxis(quat, -1, 0)
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def errors(sim: np.ndarray, im: np.ndarray, nq: int) -> dict[str, np.ndarray]:
  """|imagined - simulated| (B, H + 1) of states (B, H + 1, S): root position and
  height (cm), yaw and orientation angle (deg), mean joint angle (deg)."""
  dot = np.abs((sim[..., 3:7] * im[..., 3:7]).sum(-1)).clip(max=1.0)
  dyaw = np.angle(np.exp(1j * (yaw(im[..., 3:7]) - yaw(sim[..., 3:7]))))
  q = slice(13, 13 + nq)
  return {
    "root_cm": 100 * np.linalg.norm(im[..., :3] - sim[..., :3], axis=-1),
    "height_cm": 100 * np.abs(im[..., 2] - sim[..., 2]),
    "yaw_deg": np.degrees(np.abs(dyaw)),
    "rot_deg": np.degrees(2 * np.arccos(dot)),
    "joint_deg": np.degrees(np.abs(im[..., q] - sim[..., q]).mean(-1)),
  }


def plot(path: Path, title: str, sim, im, err: dict, frames0, joint_names) -> None:
  """Root height and two joint angles (sim solid, imagined dashed), and the root
  position, orientation and mean joint errors over the horizon, one colour per start;
  states (B, H + 1, S). Errors in orientation, not yaw: a flip's yaw jumps by 180 deg
  when the root turns upside down."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  panels = [("root height (m)", sim[..., 2], im[..., 2])]
  for name in PLOT_JOINTS:
    j = 13 + joint_names.index(name)
    panels.append((f"{name} (deg)", np.degrees(sim[..., j]), np.degrees(im[..., j])))
  panels += [
    ("root position error (cm)", None, err["root_cm"]),
    ("root orientation error (deg)", None, err["rot_deg"]),
    ("mean joint error (deg)", None, err["joint_deg"]),
  ]
  colors = plt.get_cmap("tab10").colors[: len(frames0)]
  fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, layout="constrained")
  steps = np.arange(sim.shape[1])
  for ax, (label, sim_y, im_y) in zip(axes.flat, panels, strict=True):
    for b, color in enumerate(colors):
      if sim_y is not None:
        ax.plot(steps, sim_y[b], color=color, lw=1.4)
      ax.plot(steps, im_y[b], color=color, lw=1.4, ls="-" if sim_y is None else "--")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
  for ax in axes[1]:
    ax.set_xlabel("imagined step (20 ms)")
  handles = [plt.Line2D([], [], color="k", ls=ls) for ls in ("-", "--")]
  handles += [plt.Line2D([], [], color=color) for color in colors]
  names = ["simulated", "imagined", *[f"start frame {k}" for k in frames0]]
  fig.legend(handles, names, loc="outside right center", fontsize=9)
  fig.suptitle(title, fontsize=11)
  fig.savefig(path, dpi=150)
  plt.close(fig)


def qpos_of(states: np.ndarray, nq_model: int, free_adr, joint_adr) -> np.ndarray:
  """MuJoCo qpos (..., nq_model) of states (..., S): root pose and joint angles."""
  qpos = np.zeros((*states.shape[:-1], nq_model))
  qpos[..., free_adr] = states[..., :7]
  qpos[..., joint_adr] = states[..., 13 : 13 + len(joint_adr)]
  return qpos


def project(points: np.ndarray, position: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Pixel coordinates (K, 2) of world points (K, 3) in the video camera."""
  forward = (target - position) / np.linalg.norm(target - position)
  right = np.cross(forward, (0.0, 0.0, 1.0))
  right /= np.linalg.norm(right)
  down = np.cross(forward, right)
  d = points - position
  scale = 0.5 * HEIGHT / np.tan(np.radians(CAMERA_FOV_DEG) / 2) / (d @ forward)
  return np.stack(
    [WIDTH / 2 + scale * (d @ right), HEIGHT / 2 + scale * (d @ down)], -1
  )


def annotate(frame: np.ndarray, lines: list, labels: list) -> np.ndarray:
  """``frame`` (H, W, 3) with ``lines`` [(text, rgb)] at the top-left and ``labels``
  [(xy, text, rgb)] centred under pixel xy."""
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default(size=CAPTION_PX)
  for row, (text, rgb) in enumerate(lines):
    draw.text((CAPTION_PX, CAPTION_PX * (1 + 1.4 * row)), text, font=font, fill=rgb)
  for (x, y), text, rgb in labels:
    draw.text((x, y + CAPTION_PX), text, font=font, fill=rgb, anchor="mt")
  return np.asarray(image)


def render(args, model, segments: list[dict], out_path: Path, fps: float) -> None:
  """Video of ``segments``, each with qpos ``sim`` and ``imagined`` (n, nq), camera
  positions and targets ``camera`` (n, 3), a ``title`` with a {step} field and
  per-frame ``error`` lines."""
  import imageio_ffmpeg
  import mediapy
  import mujoco
  import viser

  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  imagined_model = ghost_model_of(model, IMAGINED_RGB)
  imagined_model.mesh_vert[:] = model.mesh_vert  # a solid robot: no inset
  data = mujoco.MjData(model)
  background = Image.new("RGBA", (WIDTH, HEIGHT), (*BACKGROUND_RGB, 255))

  def frame_at(browser, scene, seg: dict, i: int) -> np.ndarray:
    data.qpos[:] = seg["sim"][i]
    mujoco.mj_kinematics(scene.mj_model, data)
    scene.clear()
    imagined = seg["imagined"][i].copy()
    imagined[:3] += SHIFT
    scene.add_ghost_mesh(imagined, imagined_model, alpha=1.0)
    scene.update_from_mjdata(data)
    # Ghosts are made without shadows; the imagined robot casts one like the other.
    for handle in scene._ghost_handles.values():
      if not handle.cast_shadow:
        handle.cast_shadow = True
    return browser.render(seg["camera"][0][i], seg["camera"][1][i], background)

  server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
  start = time.monotonic()
  with ExitStack() as stack:
    stack.callback(server.stop)
    browser = stack.enter_context(HeadlessBrowser(server, args.chrome, args.gpu))
    scene = build_scene(browser.client, model)
    video = stack.enter_context(
      mediapy.VideoWriter(out_path, shape=(HEIGHT, WIDTH), fps=fps, crf=18)
    )
    settled = False
    for seg in segments:
      last = len(seg["sim"]) - 1
      for i in range(last + 1):
        frame = frame_at(browser, scene, seg, i)
        while not settled:  # meshes load asynchronously: wait until renders agree
          time.sleep(0.5)
          again = frame_at(browser, scene, seg, i)
          settled, frame = bool(np.array_equal(again, frame)), again
        # Labels on the floor under each root, in front of the feet.
        floor = np.stack((seg["sim"][i, :3], seg["imagined"][i, :3] + SHIFT))
        floor[:, :2] += LABEL_FORWARD_M * _TOWARD_CAMERA
        floor[:, 2] = 0.0
        xy = project(floor, seg["camera"][0][i], seg["camera"][1][i])
        lines = [
          (seg["title"].format(step=i), CAPTION_RGB),
          ("left: MuJoCo simulation", CAPTION_RGB),
          (
            "right, orange: world-model imagination (same policy, same start state)",
            CAPTION_IMAGINED_RGB,
          ),
          (seg["error"][i], CAPTION_RGB),
        ]
        labels = [
          (xy[0], "simulation", CAPTION_RGB),
          (xy[1], "imagination", CAPTION_IMAGINED_RGB),
        ]
        frame = annotate(frame, lines, labels)
        hold = HOLD_FIRST_S if i == 0 else HOLD_LAST_S if i == last else 0.0
        for _ in range(1 + round(hold * fps)):
          video.add_image(frame)
  rendered = sum(len(seg["sim"]) for seg in segments)
  elapsed = time.monotonic() - start
  print(
    f"[imagine] {out_path.name}: {rendered} frames in {elapsed:.0f} s "
    f"({elapsed / rendered:.2f} s per frame)",
    flush=True,
  )


def default_starts(iface, sim: dict, horizon: int, margin: int) -> list[int]:
  """4 evenly spaced steps t0 with frame + H + margin < clip end and t0 + H <= T."""
  steps = len(sim["action"])
  frame = sim["frame"][: steps - horizon + 1]
  valid = torch.nonzero(frame + horizon + margin < iface.clip_end(frame))
  if valid.numel() == 0:
    raise SystemExit(f"the clip is too short for H = {horizon}")
  return np.linspace(0, int(valid.max()), 4).round().astype(int).tolist()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint", type=Path, help="checkpoint file model_<it>.pt")
  source.add_argument("--run-dir", type=Path, help="run dir: its last checkpoint")
  parser.add_argument("--clip", required=True, help="clip name (npz file stem)")
  parser.add_argument("--motions-dir", type=Path, help="folder of the clip")
  parser.add_argument("--horizon", type=int, action="append", help="H (default 32)")
  parser.add_argument("--starts", help="comma-separated start steps (default: 4)")
  parser.add_argument("--slowdown", type=float, default=2.0, help="video slow motion")
  parser.add_argument("--gpu", type=int, default=0, help="GPU of sim, model, WebGL")
  parser.add_argument("--chrome", help="Chromium executable (default: Playwright's)")
  parser.add_argument("--no-video", action="store_true", help="npz and PNG only")
  args = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # --gpu is nvidia-smi's index
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before the first CUDA call
  checkpoint = (args.checkpoint or latest_checkpoint(args.run_dir)).absolute()
  motion = resolve_motion(args.clip, args.motions_dir, checkpoint.parent).absolute()
  if not args.no_video:
    args.chrome = args.chrome or find_chrome()
    check_chrome(args.chrome)
  runner, sim = simulate(checkpoint, motion)
  iface, env = runner.interface, runner.env.unwrapped
  context = play_context(iface)
  robot, model = env.scene["robot"], env.sim.mj_model
  joint_names = list(robot.joint_names)
  free_adr = robot.indexing.free_joint_q_adr.cpu().numpy()
  assert np.array_equal(free_adr, np.arange(7)), "the render shifts qpos[:3]"
  joint_adr = robot.indexing.joint_q_adr.cpu().numpy()
  # The task's reward head on the sim transitions, as imagination scores its own.
  prev = torch.cat((torch.zeros_like(sim["action"][:1]), sim["action"][:-1]))
  head = iface.reward(sim["state"][1:], sim["action"], prev, sim["frame"])
  steps = len(sim["action"])
  print(f"[imagine] {checkpoint} on {motion}: {steps} steps", flush=True)
  out_dir = VIDEOS_DIR / "imagine" / args.clip
  out_dir.mkdir(parents=True, exist_ok=True)
  for horizon in args.horizon or [32]:
    margin = runner.alg.policy_cfg.seed_end_margin
    starts = (
      [int(t) for t in args.starts.split(",")]
      if args.starts
      else default_starts(iface, sim, horizon, margin)
    )
    if max(starts) + horizon > steps:
      raise SystemExit(f"start {max(starts)} + H {horizon} is beyond {steps} steps")
    im = imagine(runner, sim, starts, horizon, context)
    window = np.array(starts)[:, None] + np.arange(horizon + 1)  # (B, H + 1)
    sim_states = sim["state"].cpu().numpy()[window]
    err = errors(sim_states, im["state"], iface.nq)
    frames0 = sim["frame"].cpu().numpy()[starts]
    norm = horizon * iface.max_step_reward
    rewards = {  # sum over the H steps / (H x max step reward)
      "imagined": im["reward"].sum(1) / norm,
      "sim_head": head.cpu().numpy()[window[:, :-1]].sum(1) / norm,
      "sim_env": sim["reward"].cpu().numpy()[window[:, :-1]].sum(1) / norm,
    }
    survival = np.prod(1.0 - im["termination"], axis=1)
    np.savez(
      out_dir / f"imagine_h{horizon}.npz",
      starts=starts,
      frames0=frames0,
      sim_states=sim_states,
      imagined_states=im["state"],
      sim_actions=sim["action"].cpu().numpy()[window[:, :-1]],
      imagined_actions=im["action"],
      imagined_rewards=im["reward"],
      imagined_termination=im["termination"],
      imagined_survival=survival,
      context=context.cpu().numpy()[0],
      first_obs_error=im["obs_error"],
      first_action_error=im["action_error"],
      joint_names=joint_names,
      checkpoint=str(checkpoint),
      motion=str(motion),
      max_step_reward=iface.max_step_reward,
      **{f"error_{key}": value for key, value in err.items()},
      **{f"reward_{key}": value for key, value in rewards.items()},
    )
    print(
      f"[imagine] H {horizon}, starts {starts} (frames {frames0.tolist()}); first "
      f"observation |diff| {im['obs_error'].max():.2e}, action "
      f"{im['action_error'].max():.2e}",
      flush=True,
    )
    for h in (h for h in REPORT_STEPS if h <= horizon):
      means = ", ".join(f"{key} {value[:, h].mean():.2f}" for key, value in err.items())
      print(f"  step {h:3d}: {means}", flush=True)
    sums = ", ".join(f"{key} {value.mean():.3f}" for key, value in rewards.items())
    print(f"  reward / (H max): {sums}; survival {survival.mean():.3f}", flush=True)
    title = f"{args.clip}, H = {horizon}"
    plot(
      out_dir / f"imagine_error_h{horizon}.png",
      title,
      sim_states,
      im["state"],
      err,
      frames0.tolist(),
      joint_names,
    )
    if args.no_video:
      continue
    segments = []
    for b in range(len(starts)):
      sim_qpos = qpos_of(sim_states[b], model.nq, free_adr, joint_adr)
      _, targets = camera_path(
        {"sub_qpos": sim_qpos, "free_q_adr": free_adr, "physics_dt": env.step_dt}
      )
      targets = targets + 0.5 * SHIFT
      segments.append(
        {
          "sim": sim_qpos,
          "imagined": qpos_of(im["state"][b], model.nq, free_adr, joint_adr),
          "camera": (targets + CAMERA_DISTANCE_M * view_direction(), targets),
          "title": f"{args.clip}   start frame {frames0[b]}   step {{step:3d}} / "
          f"{horizon}   ({1 / args.slowdown:g}x speed)",
          "error": [
            f"imagined - simulated: root {e:4.1f} cm, joints {j:4.1f} deg"
            for e, j in zip(err["root_cm"][b], err["joint_deg"][b], strict=True)
          ],
        }
      )
    fps = 1.0 / (env.step_dt * args.slowdown)
    render(args, model, segments, out_dir / f"imagine_h{horizon}.mp4", fps)
  print(f"[imagine] wrote {out_dir}", flush=True)


if __name__ == "__main__":
  main()
