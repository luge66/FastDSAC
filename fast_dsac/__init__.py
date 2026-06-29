"""
FastDSAC implementations for scalable reinforcement learning with
distributional critics.
"""

from .dsact_net import (
    Actor as DSACActor,
    Critic as DSACCritic,
    GaussianQNetwork as DSACGaussianQNetwork,
)

from .fastdsac_net import (
    Actor as FastDSACActor,
    Critic as FastDSACCritic,
    GaussianQNetwork as FastDSACGaussianQNetwork,
)

from .fast_dsac_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
)

__all__ = [
    "DSACActor",
    "DSACCritic",
    "DSACGaussianQNetwork",
    "FastDSACActor",
    "FastDSACCritic",
    "FastDSACGaussianQNetwork",
    "EmpiricalNormalization",
    "RewardNormalizer",
    "SimpleReplayBuffer",
]