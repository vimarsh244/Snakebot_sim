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
        # Actor: sim-to-real deployable, ~34 dim obs → moderate network
        # goal_vector(2) + heading(2) + joint_pos(10) + joint_vel(10) + actions(10) = 34
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=False,  # disabled — obs ranges are bounded
            stochastic=True,
            init_noise_std=0.35,
            noise_std_type="log",
        ),
        # Critic: full privileged state → ~79 dim obs → larger network
        # actor(34) + body_pos(15) + body_lin_vel(15) + body_ang_vel(15) + efforts(10) = 89
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,  # disabled — obs ranges are bounded
            stochastic=False,
            init_noise_std=1.0,
        ),
        # PPO hyper-parameters
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,          # encourage exploration
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
