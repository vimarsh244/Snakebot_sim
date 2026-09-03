# Snakebot v2 model 1200 goal evaluation

Checkpoint: `logs/rsl_rl/snakebot_locomotion_v2/2026-03-10_19-34-58/model_1200.pt`

## Protocol

- Four deterministic, cardinal goal offsets at 0.60 m from the initial robot center of mass.
- Seed: `20260903`.
- 20 Hz policy control for 75 simulated seconds.
- Observation corruption, domain randomization, and external pushes disabled as in play mode.
- Automatic terminations disabled so the full approach behavior remains visible.
- Success means the robot center of mass comes within 0.10 m of the goal.

## Results

| Goal offset | Minimum distance | Final distance | Reached 0.10 m |
| --- | ---: | ---: | --- |
| +X | 0.1085 m | 0.1205 m | No |
| +Y | 0.0756 m | 0.1251 m | Yes, at 6.25 s |
| -X | 0.1164 m | 0.1212 m | No |
| -Y | 0.1148 m | 0.1226 m | No |

All four rollouts reduced the initial 0.60 m error by about 80%. The policy consistently settled roughly 0.08–0.12 m from the target, but only the +Y sample crossed the strict 0.10 m success boundary.

## Recordings

- [+X rollout](./plus_x.mp4)
- [+Y rollout](./plus_y.mp4)
- [-X rollout](./minus_x.mp4)
- [-Y rollout](./minus_y.mp4)

Machine-readable results are available in [summary.json](./summary.json).
