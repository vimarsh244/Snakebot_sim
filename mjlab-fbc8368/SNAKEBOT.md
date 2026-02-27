# Snakebot (chain_5) in mjlab

This document describes how to set up, train, and visualize the 5-module snakebot (closed-chain MJCF from `snake_description/chain_5.xml`) using mjlab.

## Model

- **Robot**: 5-module snake chain with closed kinematic loops (equality welds). Actuated joints: 10 (2 per module: `m*_Revolute-15`, `m*_Revolute-16`).
- **MJCF**: `src/mjlab/asset_zoo/robots/snakebot/xmls/chain_5.xml` (copy of the original with minimal Warp-compat edit: `noslip_iterations` removed).
- **Meshes**: All STLs live under `snakebot/xmls/meshes/`.

## RL Implementation Details

The RL pipeline uses an **asymmetric actor/critic** setup tuned for serpentine locomotion at **10 Hz**:

### Observations

**Actor** (Sim-to-real deployable — ~42 dim):
- Joint positions and velocities (10 + 10)
- Previous actions (10)
- Velocity commands (3: vx, vy, wz)
- Root module IMU (linear velocity, angular velocity, projected gravity)

**Critic** (Privileged simulation state — ~139 dim):
- All actor observations (without noise)
- `all_body_pos_rel`: Complete kinematic chain positions relative to root (15)
- `all_body_lin_vel`: Linear velocities of all 5 modules (15)
- `all_body_ang_vel`: Simulated IMU data (angular velocity) for every module (15)
- `joint_efforts`: Actuator forces for energy estimation (10)

### Rewards

Rewards are tuned for serpentine motion (no gait/foot dependence):
- **track_linear_velocity**: Primary reward for following forward command
- **snake_forward_progress**: Bonus for actual CoM forward displacement
- **snake_body_height**: Encourages modules to hover near 8 cm (normal ground contact height)
- **snake_undulation**: Rewards alternating joint signs (the classic serpentine wave pattern)
- **snake_lateral_deviation**: Penalises sideways drift
- **upright**: Soft penalty to keep the head module from fully flipping over

### Control Frequency
The environment operates at **10 Hz** (MuJoCo timestep `0.005 s` with `decimation=20`).

## Setup

From the mjlab repo root:

```bash
cd mjlab-fbc8368
uv sync
```

Requires Python 3.10–3.13 and (for GPU training) NVIDIA GPU + CUDA.

## Training

Train with PPO (wandb logging by default):

```bash
uv run train Mjlab-Velocity-Flat-Snakebot --env.scene.num-envs 2048
```

Tune parallelism:

```bash
uv run train Mjlab-Velocity-Flat-Snakebot --env.scene.num-envs 1024
uv run train Mjlab-Velocity-Flat-Snakebot --env.scene.num-envs 4096 --gpu-ids 0 1
```

Training uses **Weights & Biases** by default. Log in once:

```bash
wandb login
```

Runs appear under the `mjlab` project. Checkpoints and ONNX are saved under `logs/rsl_rl/snakebot_velocity/<timestamp>/` and (if `upload_model` is true) synced to wandb.

## Visualization

### Zero / random agent (no policy)

```bash
uv run play Mjlab-Velocity-Flat-Snakebot --agent zero
uv run play Mjlab-Velocity-Flat-Snakebot --agent random
```

### Trained policy (from wandb)

```bash
uv run play Mjlab-Velocity-Flat-Snakebot --wandb-run-path YOUR_ENTITY/mjlab/RUN_ID
```

### Trained policy (local checkpoint)

```bash
uv run play Mjlab-Velocity-Flat-Snakebot --checkpoint-file path/to/model_XXXX.pt
```

### Viewers

- **Native MuJoCo viewer** (default): `uv run play ...`
- **Viser (web)**: `uv run play ... --viewer viser` — opens in browser (e.g. http://localhost:8012).

### Recording video

```bash
uv run play Mjlab-Velocity-Flat-Snakebot --agent zero --video
```

Videos are written under the run directory.

## Wandb

- **Project**: `mjlab` (set in task RL config).
- **Tags**: Add via `--agent.wandb-tags tag1 tag2`.
- **Resume**: `uv run train Mjlab-Velocity-Flat-Snakebot --agent.resume --wandb-run-path ENTITY/mjlab/RUN_ID`.

## If MuJoCo Warp fails

The snake uses equality constraints (welds). If training fails on GPU (e.g. Warp constraint limits), you can:

1. Reduce `--env.scene.num-envs` (e.g. 512).
2. Try CPU-only: set `CUDA_VISIBLE_DEVICES=""` and run training (slower).
3. Use the original `snake_description/chain_5.xml` (with `noslip_iterations`) in standard MuJoCo for visualization; the mjlab copy only removes `noslip_iterations` for Warp compatibility.

## File layout

- `src/mjlab/asset_zoo/robots/snakebot/` — robot assets and config
  - `xmls/chain_5.xml` — MJCF (Warp-compat)
  - `xmls/meshes/*.stl` — meshes
  - `snakebot_constants.py` — `get_snakebot_robot_cfg()`, action scale
- `src/mjlab/tasks/velocity/config/snakebot/` — task registration
  - `env_cfgs.py` — flat velocity env, decoupled actor/critic definition (10Hz)
  - `snakebot_mdp.py` — custom serpentine rewards and multi-module observation functions
  - `rl_cfg.py` — PPO + wandb configs (large actor/critic networks)
  - `__init__.py` — registers `Mjlab-Velocity-Flat-Snakebot`
