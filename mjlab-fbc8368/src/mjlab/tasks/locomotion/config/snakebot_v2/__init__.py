"""Register snakebot locomotion v2 task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import snakebot_locomotion_v2_flat_cfg
from .rl_cfg import snakebot_locomotion_v2_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Locomotion-Flat-Snakebot-v2",
  env_cfg=snakebot_locomotion_v2_flat_cfg(),
  play_env_cfg=snakebot_locomotion_v2_flat_cfg(play=True),
  rl_cfg=snakebot_locomotion_v2_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
