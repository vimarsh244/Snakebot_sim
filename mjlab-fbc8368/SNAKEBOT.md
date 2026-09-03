# Snakebot (chain_5) in mjlab

This document describes how to set up, train, and visualize the 5-module snakebot (closed-chain MJCF from `snake_description/chain_5.xml`) using mjlab.

## Model

- **Robot**: 5-module snake chain with closed kinematic loops (equality welds). Actuated joints: 10 (2 per module: `m*_Revolute-15`, `m*_Revolute-16`).
- **MJCF**: `src/mjlab/asset_zoo/robots/snakebot/xmls/chain_5.xml` (copy of the original with Warp-compat edits: `noslip_iterations` removed, equality constraint `solref` relaxed to `0.002 1`).
- **Meshes**: All STLs live under `snakebot/xmls/meshes/`.

## Available Tasks

### 1. `Mjlab-Velocity-Flat-Snakebot` — Velocity Tracking

Forward velocity tracking task. The robot learns to follow velocity commands.

**Actor** (~44 dim): Phase clock (2), actuated joint pos/vel (20), last action (10), velocity commands (3), root IMU (9)  
**Critic** (~99 dim): Actor obs + per-module pos/vel/ang-vel (45) + joint efforts (10)

**Rewards** (literature: Bing 2020, Shi 2020, Qiu 2021, Singh 2022):
| Term | Weight | Purpose |
|---|---|---|
| `forward_velocity` | +6.0 | World +X CoM velocity (primary drive) |
| `alive_bonus` | +0.5 | Keep-alive baseline |
| `lateral_velocity` | -0.05 | Penalise Y drift |
| `vertical_velocity` | -0.05 | Penalise vertical bounce |
| `yaw_rate` | -0.02 | Penalise spin-in-place |
| `action_smoothness` | -0.01 | Smooth gait → hardware transfer |
| `control_cost` | -0.005 | Energy efficiency |
| `dof_pos_limits` | -0.002 | Soft joint-limit regularisation |

### 2. `Mjlab-Locomotion-Flat-Snakebot` — Goal-Reaching

Navigate to a random XY goal 0.3–0.8 m away. Reward design informed by 5 research papers (COBRA thesis, snakebot-gym, Naish/EELS, serpentine locomotion RL, sensors-22-09867).

**Actor** (~36 dim, hardware-deployable):
- Phase clock (2): sin/cos of normalised episode time for gait coordination
- Goal vector in body frame (2): forward/lateral offset to goal via full quaternion rotation
- Heading to goal (2): sin/cos of angle between heading and goal direction
- Joint positions/velocities (20)
- Previous actions (10)

**Critic** (~91 dim, privileged):
- All actor obs + per-module positions/velocities/angular-vel (45) + joint efforts (10)

**Rewards** (COBRA thesis, snakebot-gym, Naish/EELS, Zhang 2024):
| Term | Weight | Purpose |
|---|---|---|
| `progress_reward` | +25.0 | Δ distance to goal (KEY signal) |
| `heading_alignment` | +0.5 | Face the goal (cosine, Naish/EELS) |
| `goal_reached_bonus` | +40.0 | Sparse bonus on arrival (<25 cm) |
| `distance_penalty` | +3.0 | Dense proximity shaping (higher when closer) |
| `alive_bonus` | +0.3 | Keep-alive baseline |
| `action_smoothness` | -0.01 | Smooth gait → hardware transfer |
| `control_cost` | -0.005 | Energy efficiency |
| `dof_pos_limits` | -0.002 | Soft joint-limit regularisation |

**Episode**: 90 s max (1800 steps at 10 Hz). Terminates early on goal reach.

## Spawning and Physics

### Spawning orientation

The snake's MJCF wraps all bodies in `<frame euler="0 1.5707963267948966 0">` — a 90° rotation about world Y — so 
the chain lies flat along world +X with skin geoms contacting the ground plane. `INIT_STATE.rot` in 
`snakebot_constants.py` must encode this same rotation as a quaternion `(0.7071068, 0.0, 0.7071068, 0.0)`. 
Without it `reset_root_state_uniform` rewrites the root quaternion to identity and the snake spawns vertically 
(chain axis along world +Z).

### Solver settings

The snake uses equality-weld closed kinematic chains which require a stiff Newton solver. Both tasks 
override the base factory's defaults:

| Parameter | Base default | Snakebot override |
|---|---|---|
| `timestep` | 0.005 s | 0.005 s (same) |
| `iterations` | 10 | **50** |
| `ls_iterations` | 20 | **30** |
| `impratio` | 1.0 | **10.0** |
| `decimation` | 4 | **20** (→ 10 Hz RL) |

For initial learning stability, interval perturbation events (`push_robot`, external wrench pulses) are
disabled in both snake tasks and only moderate reset/startup domain randomization is applied.

## Setup

From the mjlab repo root:

```bash
cd mjlab-fbc8368
git submodule update --init --recursive
uv sync
```

Requires Python 3.10–3.13 and (for GPU training) NVIDIA GPU + CUDA.

The browser viewer uses the pinned `third_party/mjswan` submodule from the
[`vimarsh244/mjswan`](https://github.com/vimarsh244/mjswan) fork. For a fresh
checkout, either clone this repository with `--recurse-submodules` or run the
submodule command above before using the mjswan viewer.

## Training

### Velocity task
```bash
uv run train Mjlab-Velocity-Flat-Snakebot --env.scene.num-envs 2048
```

### Locomotion task (goal-reaching)
```bash
uv run train Mjlab-Locomotion-Flat-Snakebot --env.scene.num-envs 2048
```

### Locomotion v2 task (goal-reaching + side-goal preference)
```bash
uv run train Mjlab-Locomotion-Flat-Snakebot-v2 --env.scene.num-envs 2048
```

Current v2 defaults:
- Goal distance curriculum expanded to farther targets (`0.50` to `3.00` m).
- Episode timeout increased to `75` s to improve goal completion on farther targets.
- Goal-reaching terms are weighted slightly higher (`progress`, `velocity_to_goal`, `distance_shaping`, `goal_reached_bonus`).
- Heading-alignment reward is removed in v2, with added COM-to-goal velocity reward to favor direct lateral travel (sidewinding-friendly).
- In `--viewer viser`, the goal is shown as a red debug dot (enable debug visualization in the viewer controls).

Tune parallelism:
```bash
uv run train Mjlab-Locomotion-Flat-Snakebot --env.scene.num-envs 1024
uv run train Mjlab-Locomotion-Flat-Snakebot --env.scene.num-envs 4096 --gpu-ids 0 1
```

Training uses **Weights & Biases** by default. Log in once: `wandb login`.

Runs appear under the `mjlab` project. Checkpoints and ONNX are saved under `logs/rsl_rl/<experiment_name>/<timestamp>/`.

## Visualization

### Zero / random agent (no policy)

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot --agent zero
uv run play Mjlab-Locomotion-Flat-Snakebot --agent random
```

### Trained policy (from wandb)

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot --wandb-run-path YOUR_ENTITY/mjlab/RUN_ID
```

### Trained policy (local checkpoint)

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot --checkpoint-file path/to/model_XXXX.pt
```

### Viewers

- **Native MuJoCo viewer** (default): `uv run play ...`
- **Viser (web)**: `uv run play ... --viewer viser` — opens in browser (e.g. http://localhost:8012).
- **mjswan (web)**: `uv run play Mjlab-Locomotion-Flat-Snakebot-v2 --viewer mjswan`

The mjswan viewer auto-selects the best local Snakebot v2 ONNX export. To select
one explicitly, run:

```bash
uv run snakebot_mjswan --onnx-file logs/rsl_rl/snakebot_locomotion_v2/RUN/RUN.onnx
```

### Recording video

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot --agent zero --video
```

## Wandb

- **Project**: `mjlab` (set in task RL config).
- **Tags**: Add via `--agent.wandb-tags tag1 tag2`.
- **Resume**: `uv run train Mjlab-Locomotion-Flat-Snakebot --agent.resume --wandb-run-path ENTITY/mjlab/RUN_ID`.

## If MuJoCo Warp fails

The snake uses equality constraints (welds). If training fails on GPU (e.g. Warp constraint limits), you can:

1. Reduce `--env.scene.num-envs` (e.g. 512).
2. Try CPU-only: set `CUDA_VISIBLE_DEVICES=""` and run training (slower).
3. Use the original `snake_description/chain_5.xml` (with `noslip_iterations`) in standard MuJoCo for visualization; the mjlab copy only removes `noslip_iterations` for Warp compatibility.

## File layout

- `src/mjlab/asset_zoo/robots/snakebot/` — robot assets and config
  - `xmls/chain_5.xml` — MJCF (Warp-compat, relaxed solref)
  - `xmls/meshes/*.stl` — meshes
  - `snakebot_constants.py` — `get_snakebot_robot_cfg()`, action scale
- `src/mjlab/tasks/velocity/config/snakebot/` — velocity task
  - `env_cfgs.py` — flat velocity env (10 Hz, asymmetric actor/critic)
  - `snakebot_mdp.py` — velocity reward & observation functions
  - `rl_cfg.py` — PPO + wandb configs
  - `__init__.py` — registers `Mjlab-Velocity-Flat-Snakebot`
- `src/mjlab/tasks/locomotion/config/snakebot/` — locomotion task (NEW)
  - `env_cfg.py` — goal-reaching env (10 Hz, goal sampling, distance rewards)
  - `locomotion_mdp.py` — goal-reaching rewards & observations
  - `rl_cfg.py` — PPO config for goal-reaching
  - `__init__.py` — registers `Mjlab-Locomotion-Flat-Snakebot`