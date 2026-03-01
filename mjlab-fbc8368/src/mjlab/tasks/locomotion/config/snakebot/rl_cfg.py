"""RL configuration for Snakebot goal-reaching locomotion task.

Asymmetric actor-critic PPO config optimised for 10 Hz snake locomotion
with goal-reaching reward structure.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def snakebot_locomotion_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Snakebot locomotion task."""
    return RslRlOnPolicyRunnerCfg(
        # Actor: ~36 dim obs (added phase_clock) → larger network for better capacity
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=1.0,
            noise_std_type="log",
        ),
        # Critic: ~91 dim privileged obs → large network
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
            stochastic=False,
            init_noise_std=1.0,
        ),
        # PPO hyper-parameters — higher entropy for exploration
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.02,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="snakebot_locomotion",
        logger="wandb",
        wandb_project="mjlab",
        save_interval=100,
        num_steps_per_env=24,           # 24 steps × 0.1 s = 2.4 s rollout
        max_iterations=10_000,
    )
