"""RL configuration for snakebot locomotion v2."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def snakebot_locomotion_v2_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a robust PPO config for snakebot locomotion v2."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=0.25,
      noise_std_type="log",
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="snakebot_locomotion_v2",
    logger="wandb",
    wandb_project="mjlab",
    save_interval=100,
    num_steps_per_env=64,
    clip_actions=0.9,
    max_iterations=15_000,
  )
