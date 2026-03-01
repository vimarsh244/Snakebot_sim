"""RL configuration for Snakebot velocity task.

Actor and critic use different network sizes reflecting their very different
observation dimensions (~44 vs ~99 dims after asymmetric split).
obs_normalization is enabled on both to handle the wide input ranges from
per-module position / velocity data.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def snakebot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Snakebot velocity task."""
    return RslRlOnPolicyRunnerCfg(
        # ------------------------------------------------------------------
        # Actor: sim-to-real deployable, small obs → moderate network
        # ------------------------------------------------------------------
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,   # normalise ~44-dim actor obs
            stochastic=True,
            init_noise_std=1.0,
            noise_std_type="log",
        ),
        # ------------------------------------------------------------------
        # Critic: full privileged state → larger network
        # ------------------------------------------------------------------
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,   # normalise ~99-dim critic obs
            stochastic=False,
            init_noise_std=1.0,
        ),
        # ------------------------------------------------------------------
        # PPO hyper-parameters tuned for 10 Hz snake locomotion
        # ------------------------------------------------------------------
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,    # slightly lower for larger critic
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="snakebot_velocity",
        logger="wandb",
        wandb_project="mjlab",
        save_interval=100,
        num_steps_per_env=24,        # 24 steps × 0.1 s = 2.4 s rollout at 10 Hz
        max_iterations=10_000,
    )
