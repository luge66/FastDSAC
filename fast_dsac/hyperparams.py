from dataclasses import dataclass

import tyro


@dataclass
class BaseArgs:
    # Experiment
    env_name: str = "h1hand-stand-v0"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    device_rank: int = 0
    exp_name: str = "FastDSAC"
    project: str = "FastDSAC"
    use_wandb: bool = True
    checkpoint_path: str | None = None

    # Environment and training
    num_envs: int = 128
    num_eval_envs: int = 128
    total_timesteps: int = 100000
    learning_starts: int = 10
    num_updates: int = 2
    policy_frequency: int = 2
    batch_size: int = 32768
    buffer_size: int = 1024 * 50
    num_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.1

    # Optimizer
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate_end: float = 3e-4
    actor_learning_rate_end: float = 3e-4
    weight_decay: float = 0.1

    # Actor
    std_min: float = 0.001
    std_max: float = 0.4
    log_std_init: float = 0.0

    # Mean-centered truncation radius used by FastDSAC
    policy_noise: float = 0.001

    use_nact: bool = True
    target_entropy: float = 0.0

    # Critic
    v_min: float = -250.0
    v_max: float = 250.0
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    use_cdq: bool = True
    bias: float = 0.1

    # Normalization and optimization options
    obs_normalization: bool = True
    reward_normalization: bool = False
    use_grad_norm_clipping: bool = False
    max_grad_norm: float = 0.0
    disable_bootstrap: bool = False
    amp: bool = True
    amp_dtype: str = "bf16"
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    # MuJoCo Playground options
    use_domain_randomization: bool = False
    use_push_randomization: bool = False
    use_tuned_reward: bool = False

    # Evaluation and saving
    measure_burnin: int = 3
    eval_interval: int = 1000
    render_interval: int = 5000
    save_interval: int = 20000


def get_args(method: str) -> BaseArgs:
    """Parse task-specific arguments for DSAC or FastDSAC."""
    method = method.lower()
    if method not in {"dsac", "fastdsac"}:
        raise ValueError(
            f"Unsupported method: {method}. Expected 'dsac' or 'fastdsac'."
        )

    base_args = tyro.cli(BaseArgs)

    common_env_to_args_class = {
        # HumanoidBench
        "h1hand-run-v0": H1HandRunArgs,
        "h1hand-walk-v0": H1HandWalkArgs,
        "h1hand-stand-v0": H1HandStandArgs,
        "h1hand-pole-v0": H1HandPoleArgs,
        "h1hand-slide-v0": H1HandSlideArgs,
        "h1hand-hurdle-v0": H1HandHurdleArgs,
        # MuJoCo Playground
        "G1JoystickFlatTerrain": G1JoystickFlatTerrainArgs,
        "G1JoystickRoughTerrain": G1JoystickRoughTerrainArgs,
        "T1JoystickFlatTerrain": T1JoystickFlatTerrainArgs,
        "T1JoystickRoughTerrain": T1JoystickRoughTerrainArgs,
    }

    fastdsac_env_to_args_class = {
        "G1JoystickFlatTerrain": FastDSACG1JoystickFlatTerrainArgs,
        "G1JoystickRoughTerrain": FastDSACG1JoystickRoughTerrainArgs,
        "T1JoystickFlatTerrain": FastDSACT1JoystickFlatTerrainArgs,
        "T1JoystickRoughTerrain": FastDSACT1JoystickRoughTerrainArgs,
    }

    if method == "fastdsac" and base_args.env_name in fastdsac_env_to_args_class:
        args_class = fastdsac_env_to_args_class[base_args.env_name]
    else:
        args_class = common_env_to_args_class.get(base_args.env_name)

    if args_class is not None:
        return tyro.cli(args_class)

    if base_args.env_name.startswith(("h1hand-", "h1-")):
        return tyro.cli(HumanoidBenchArgs)

    return tyro.cli(MuJoCoPlaygroundArgs)


@dataclass
class HumanoidBenchArgs(BaseArgs):
    total_timesteps: int = 100000
    log_std_init: float = -2.0


@dataclass
class H1HandRunArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-run-v0"
    total_timesteps: int = 100000


@dataclass
class H1HandWalkArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-walk-v0"
    total_timesteps: int = 100000


@dataclass
class H1HandStandArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-stand-v0"
    total_timesteps: int = 100000


@dataclass
class H1HandPoleArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-pole-v0"
    total_timesteps: int = 150000


@dataclass
class H1HandSlideArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-slide-v0"
    total_timesteps: int = 250000


@dataclass
class H1HandHurdleArgs(HumanoidBenchArgs):
    env_name: str = "h1hand-hurdle-v0"
    total_timesteps: int = 300000


@dataclass
class MuJoCoPlaygroundArgs(BaseArgs):
    v_min: float = -10.0
    v_max: float = 10.0
    buffer_size: int = 1024 * 10
    num_envs: int = 1024
    num_eval_envs: int = 1024
    gamma: float = 0.97


@dataclass
class G1JoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "G1JoystickFlatTerrain"
    total_timesteps: int = 100000


@dataclass
class G1JoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "G1JoystickRoughTerrain"
    total_timesteps: int = 100000


@dataclass
class T1JoystickFlatTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1JoystickFlatTerrain"
    total_timesteps: int = 100000


@dataclass
class T1JoystickRoughTerrainArgs(MuJoCoPlaygroundArgs):
    env_name: str = "T1JoystickRoughTerrain"
    total_timesteps: int = 100000


@dataclass
class FastDSACG1JoystickFlatTerrainArgs(G1JoystickFlatTerrainArgs):
    bias: float = 1e-3
    policy_noise: float = 1e-3


@dataclass
class FastDSACG1JoystickRoughTerrainArgs(G1JoystickRoughTerrainArgs):
    bias: float = 1e-3
    policy_noise: float = 1e-2


@dataclass
class FastDSACT1JoystickFlatTerrainArgs(T1JoystickFlatTerrainArgs):
    bias: float = 0.1
    policy_noise: float = 1e-3


@dataclass
class FastDSACT1JoystickRoughTerrainArgs(T1JoystickRoughTerrainArgs):
    bias: float = 0.1
    policy_noise: float = 1e-2
