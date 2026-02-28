# Snakebot (chain_5) in mjlab

This document describes how to set up, train, and visualize the 5-module snakebot (closed-chain MJCF from `snake_description/chain_5.xml`) using mjlab.

## Model

- **Robot**: 5-module snake chain with closed kinematic loops (equality welds). Actuated joints: 10 (2 per module: `m*_Revolute-15`, `m*_Revolute-16`).
- **MJCF**: `src/mjlab/asset_zoo/robots/snakebot/xmls/chain_5.xml` (copy of the original with Warp-compat edits: `noslip_iterations` removed, equality constraint `solref` relaxed to `0.002 1`).
- **Meshes**: All STLs live under `snakebot/xmls/meshes/`.

## Available Tasks

### 1. `Mjlab-Velocity-Flat-Snakebot` — Velocity Tracking

Forward velocity tracking task. The robot learns to follow velocity commands.

**Actor** (~42 dim): Joint pos/vel (20), last action (10), velocity commands (3), root IMU (9)
**Critic** (~139 dim): Actor obs + per-module pos/vel/ang-vel (45) + joint efforts (10)

### 2. `Mjlab-Locomotion-Flat-Snakebot` — Goal-Reaching (NEW)

Navigate to a random XY goal 1–2 m away. Reward design informed by 5 research papers (COBRA thesis, snakebot-gym, Naish/EELS, serpentine locomotion RL, sensors-22-09867).

**Actor** (~34 dim, hardware-deployable):
- Goal vector in body frame (2): XY offset to goal, rotated by robot yaw
- Heading to goal (2): sin/cos of angle between heading and goal direction
- Joint positions/velocities (20)
- Previous actions (10)

**Critic** (~89 dim, privileged):
- All actor obs + per-module positions/velocities/angular-vel (45) + joint efforts (10)

**Rewards**:
| Term | Weight | Purpose |
|---|---|---|
| `progress_reward` | +5.0 | Δ distance to goal (KEY signal) |
| `distance_penalty` | -0.5 | Persistent pull toward goal |
| `heading_alignment` | +1.0 | Face the goal (cos similarity) |
| `goal_reached_bonus` | +100.0 | Sparse bonus on arrival (<15 cm) |
| `alive_bonus` | +0.5 | Encourage long episodes |
| `control_cost` | -0.05 | Energy efficiency |
| `action_smoothness` | -0.02 | Sim-to-real, no jitter |
| `dof_pos_limits` | -1.0 | Prevent self-collision |

**Episode**: 30 s max (300 steps at 10 Hz). Terminates early on goal reach or fall-over.

## Setup

From the mjlab repo root:

```bash
cd mjlab-fbc8368
uv sync
```

Requires Python 3.10–3.13 and (for GPU training) NVIDIA GPU + CUDA.

## Training

### Velocity task
```bash
uv run train Mjlab-Velocity-Flat-Snakebot --env.scene.num-envs 2048
```

### Locomotion task (goal-reaching)
```bash
uv run train Mjlab-Locomotion-Flat-Snakebot --env.scene.num-envs 2048
```

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