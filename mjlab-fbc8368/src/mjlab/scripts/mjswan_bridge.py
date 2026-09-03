"""Snakebot locomotion v2 integration with local mjswan clone.

Builds a browser app from a snakebot v2 ONNX policy and MuJoCo scene, and can
auto-select the best local ONNX artifact from training logs.
"""

from __future__ import annotations

import json
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import tyro
from tensorboard.backend.event_processing import event_accumulator

import mjlab
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

SNAKEBOT_V2_TASK_ID = "Mjlab-Locomotion-Flat-Snakebot-v2"
MODULE_BODY_NAMES = [f"m{i}_bottom-base-plate-v1" for i in range(1, 6)]


@dataclass(frozen=True)
class BestRunInfo:
  run_dir: Path
  onnx_path: Path
  reward: float | None
  step: int | None


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def _import_local_mjswan(repo_root: Path):
  mjswan_src = repo_root / "third_party" / "mjswan" / "src"
  if not mjswan_src.exists():
    raise FileNotFoundError(
      f"Local mjswan clone not found. Expected: {repo_root / 'third_party' / 'mjswan'}"
    )
  mjswan_src_str = str(mjswan_src)
  if mjswan_src_str not in sys.path:
    sys.path.insert(0, mjswan_src_str)
  import mjswan  # type: ignore

  return mjswan


def _csv_to_list(text: str) -> list[str]:
  return [part.strip() for part in text.split(",") if part.strip()]


def _csv_to_float_list(text: str) -> list[float]:
  values: list[float] = []
  for part in _csv_to_list(text):
    try:
      values.append(float(part))
    except ValueError:
      continue
  return values


def _read_best_mean_reward(event_file: Path) -> tuple[float | None, int | None]:
  try:
    acc = event_accumulator.EventAccumulator(
      str(event_file), size_guidance={"scalars": 0}
    )
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    if "Train/mean_reward" not in tags:
      return None, None
    values = acc.Scalars("Train/mean_reward")
    if not values:
      return None, None
    best = max(values, key=lambda x: x.value)
    return float(best.value), int(best.step)
  except Exception:
    return None, None


def select_best_snakebot_v2_run(log_root: Path | None = None) -> BestRunInfo:
  log_root = log_root or (_repo_root() / "logs" / "rsl_rl" / "snakebot_locomotion_v2")
  if not log_root.exists():
    raise FileNotFoundError(f"Log root not found: {log_root}")

  candidates: list[BestRunInfo] = []
  for run_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
    onnx_files = sorted(run_dir.glob("*.onnx"))
    if not onnx_files:
      continue
    onnx_path = onnx_files[0]
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    reward: float | None = None
    step: int | None = None
    if event_files:
      reward, step = _read_best_mean_reward(event_files[0])
    candidates.append(
      BestRunInfo(
        run_dir=run_dir,
        onnx_path=onnx_path,
        reward=reward,
        step=step,
      )
    )

  if not candidates:
    raise FileNotFoundError(
      f"No ONNX files found under snakebot v2 log root: {log_root}"
    )

  def _score(item: BestRunInfo) -> tuple[int, float, float]:
    has_reward = 1 if item.reward is not None else 0
    reward_value = item.reward if item.reward is not None else float("-inf")
    mtime = item.onnx_path.stat().st_mtime
    return (has_reward, reward_value, mtime)

  return max(candidates, key=_score)


def resolve_onnx_for_checkpoint(checkpoint_path: Path) -> Path | None:
  checkpoint_path = checkpoint_path.resolve()
  if not checkpoint_path.exists():
    return None

  if checkpoint_path.suffix.lower() == ".onnx":
    return checkpoint_path

  checkpoint_dir = checkpoint_path.parent
  explicit_candidates = [
    checkpoint_path.with_suffix(".onnx"),
    checkpoint_dir / f"{checkpoint_dir.name}.onnx",
  ]

  for candidate in explicit_candidates:
    if candidate.exists():
      return candidate

  discovered = sorted(
    checkpoint_dir.glob("*.onnx"), key=lambda path: path.stat().st_mtime, reverse=True
  )
  if discovered:
    return discovered[0]

  return None


def _local_ipv4_addrs() -> list[str]:
  addrs: set[str] = set()

  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
      sock.connect(("8.8.8.8", 80))
      addr = sock.getsockname()[0]
      if addr and addr != "127.0.0.1":
        addrs.add(addr)
  except Exception:
    pass

  try:
    addrs_info = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    for item in addrs_info:
      addr = item[4][0]
      if isinstance(addr, str) and addr and not addr.startswith("127."):
        addrs.add(addr)
  except Exception:
    pass

  ordered = sorted(addrs)
  return ordered


def _infer_action_dim(model: onnx.ModelProto, fallback: int) -> int:
  if model.graph.output:
    out = model.graph.output[0]
    dims = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    if len(dims) >= 2 and dims[1] > 0:
      return int(dims[1])
  return fallback


def _normalize_length(
  values: list[float], length: int, fill: float = 0.0
) -> list[float]:
  if len(values) >= length:
    return values[:length]
  return values + [fill] * (length - len(values))


def _select_policy_joint_names(
  all_joint_names: list[str], action_dim: int
) -> list[str]:
  actuated = [name for name in all_joint_names if re.search(r"Revolute-(15|16)$", name)]
  if len(actuated) == action_dim:
    return actuated
  if len(all_joint_names) >= action_dim:
    return all_joint_names[:action_dim]
  raise ValueError(
    f"Unable to derive {action_dim} policy joints from metadata (only "
    f"{len(all_joint_names)} names available)."
  )


def _subset_defaults(
  all_joint_names: list[str],
  default_joint_pos: list[float],
  policy_joint_names: list[str],
) -> list[float]:
  if len(default_joint_pos) < len(all_joint_names):
    return [0.0] * len(policy_joint_names)

  index_by_name = {name: idx for idx, name in enumerate(all_joint_names)}
  values: list[float] = []
  for name in policy_joint_names:
    idx = index_by_name.get(name)
    if idx is None or idx >= len(default_joint_pos):
      values.append(0.0)
    else:
      values.append(float(default_joint_pos[idx]))
  return values


def build_snakebot_v2_policy_config(
  onnx_path: Path,
  output_path: Path,
) -> Path:
  model = onnx.load(str(onnx_path))
  metadata = {entry.key: entry.value for entry in model.metadata_props}

  all_joint_names = _csv_to_list(metadata.get("joint_names", ""))
  action_scale = _csv_to_float_list(metadata.get("action_scale", ""))
  stiffness = _csv_to_float_list(metadata.get("joint_stiffness", ""))
  damping = _csv_to_float_list(metadata.get("joint_damping", ""))
  default_all = _csv_to_float_list(metadata.get("default_joint_pos", ""))

  fallback_dim = len(action_scale) if action_scale else 10
  action_dim = _infer_action_dim(model, fallback=fallback_dim)
  policy_joint_names = _select_policy_joint_names(all_joint_names, action_dim)

  default_joint_pos = _subset_defaults(
    all_joint_names=all_joint_names,
    default_joint_pos=default_all,
    policy_joint_names=policy_joint_names,
  )

  action_scale = _normalize_length(action_scale, action_dim, fill=0.1)
  stiffness = _normalize_length(stiffness, action_dim, fill=12.0)
  damping = _normalize_length(damping, action_dim, fill=0.35)

  input_name = model.graph.input[0].name if model.graph.input else "obs"
  output_name = model.graph.output[0].name if model.graph.output else "actions"

  config: dict[str, Any] = {
    "policy_module": "locomotion",
    "policy_joint_names": policy_joint_names,
    "default_joint_pos": default_joint_pos,
    "action_scale": action_scale,
    "stiffness": stiffness,
    "damping": damping,
    "control_type": "joint_position",
    "obs_config": {
      input_name: [
        {
          "name": "PhaseClock",
          "period_steps": 30,
        },
        {
          "name": "GoalVectorBodyFrame",
          "command_name": "goal",
        },
        {
          "name": "HeadingToGoal",
          "command_name": "goal",
        },
        {
          "name": "JointPos",
          "history_steps": 3,
          "subtract_default": False,
        },
        {
          "name": "JointVelocities",
          "history_steps": 3,
        },
        {
          "name": "PrevActions",
          "history_steps": 3,
        },
        {
          "name": "BodyLinearVelocityRootFrame",
          "body_names": MODULE_BODY_NAMES,
          "clip": 20.0,
        },
        {
          "name": "BodyAngularVelocityRootFrame",
          "body_names": MODULE_BODY_NAMES,
          "clip": 50.0,
        },
      ]
    },
    "onnx": {
      "meta": {
        "in_keys": [input_name],
        "out_keys": [output_name],
      }
    },
    "commands": {
      "goal": {
        "inputs": [
          {
            "type": "slider",
            "name": "goal_x",
            "label": "Goal X (world)",
            "min": -2.0,
            "max": 2.0,
            "step": 0.01,
            "default": 0.8,
          },
          {
            "type": "slider",
            "name": "goal_y",
            "label": "Goal Y (world)",
            "min": -2.0,
            "max": 2.0,
            "step": 0.01,
            "default": 0.0,
          },
        ]
      }
    },
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

  return output_path


def build_snakebot_v2_mjswan_app(
  task_id: str,
  onnx_path: Path,
  output_dir: Path,
):
  if task_id != SNAKEBOT_V2_TASK_ID:
    raise ValueError(
      f"mjswan bridge currently supports only {SNAKEBOT_V2_TASK_ID}; got {task_id}"
    )

  repo_root = _repo_root()
  mjswan = _import_local_mjswan(repo_root)

  onnx_path = onnx_path.resolve()
  if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

  policy_cfg_path = onnx_path.with_suffix(".mjswan.json")
  build_snakebot_v2_policy_config(onnx_path=onnx_path, output_path=policy_cfg_path)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device="cpu")

  builder = mjswan.Builder()
  project = builder.add_project(name="Snakebot Locomotion v2")
  scene_handle = project.add_scene(name=task_id, spec=scene.spec)
  scene_handle.add_policy(
    policy=onnx.load(str(onnx_path)),
    name=onnx_path.stem,
    config_path=str(policy_cfg_path),
  )

  app = builder.build(output_dir=output_dir)
  return app, policy_cfg_path


def launch_snakebot_v2_mjswan(
  task_id: str = SNAKEBOT_V2_TASK_ID,
  onnx_path: Path | None = None,
  output_dir: Path | None = None,
  host: str = "0.0.0.0",
  port: int = 8013,
  open_browser: bool = True,
) -> None:
  output_dir = output_dir or (_repo_root() / "logs" / "mjswan" / "snakebot_v2")

  if onnx_path is None:
    best = select_best_snakebot_v2_run()
    onnx_path = best.onnx_path
    reward_text = f"{best.reward:.6f}" if best.reward is not None else "N/A"
    print(
      "[INFO] Selected best snakebot v2 ONNX: "
      f"{onnx_path} (reward={reward_text}, step={best.step})"
    )
  else:
    onnx_path = onnx_path.resolve()

  app, policy_cfg_path = build_snakebot_v2_mjswan_app(
    task_id=task_id,
    onnx_path=onnx_path,
    output_dir=output_dir.resolve(),
  )
  print(f"[INFO] Generated policy config: {policy_cfg_path}")
  print(f"[INFO] Built mjswan app at: {output_dir.resolve()}")

  if host in {"0.0.0.0", "::"}:
    addrs = _local_ipv4_addrs()
    if addrs:
      print("[INFO] LAN access URLs:")
      for addr in addrs:
        print(f"  http://{addr}:{port}")
    else:
      print(f"[INFO] LAN access enabled on port {port} (host={host}).")

  app.launch(host=host, port=port, open_browser=open_browser)


@dataclass(frozen=True)
class SnakebotMjswanConfig:
  task_id: str = SNAKEBOT_V2_TASK_ID
  onnx_file: str | None = None
  output_dir: str = "logs/mjswan/snakebot_v2"
  host: str = "0.0.0.0"
  port: int = 8013
  open_browser: bool = True


def main() -> None:
  args = tyro.cli(SnakebotMjswanConfig, config=mjlab.TYRO_FLAGS)
  launch_snakebot_v2_mjswan(
    task_id=args.task_id,
    onnx_path=Path(args.onnx_file).resolve() if args.onnx_file else None,
    output_dir=Path(args.output_dir),
    host=args.host,
    port=args.port,
    open_browser=args.open_browser,
  )


if __name__ == "__main__":
  main()
