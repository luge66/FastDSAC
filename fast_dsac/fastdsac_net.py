import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


class GaussDistribution:

    def __init__(self, mean, std, device):
        self.mean = mean
        self.std = std
        self.gauss_distribution = Independent(
            base_distribution=Normal(self.mean, self.std), reinterpreted_batch_ndims=1
        )
        self.device = device
        self.act_high_lim = torch.tensor([1.0]).to(device)
        self.act_low_lim = torch.tensor([-1.0]).to(device)

    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return (action, log_prob)

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return (action, log_prob)

    def rsample_target(self, clip):
        clip = torch.tensor([clip], dtype=torch.float32).to(self.device)
        action = self.gauss_distribution.rsample()
        action_limited = self.mean + clip * torch.tanh(action - self.mean)
        log_prob = self.gauss_distribution.log_prob(action) + (
            2 * (action - self.mean)
            + 2 * F.softplus(-2 * (action - self.mean))
            - torch.log(torch.tensor(4.0, dtype=torch.float32))
            - torch.log(clip)
        ).sum(-1)
        return action_limited, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussianQNetwork(nn.Module):

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        v_min: float,
        v_max: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.GELU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )
        self.std = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        mu = self.mu(x)
        mu = torch.clamp(mu, self.v_min, self.v_max)
        std = self.std(x)
        std = torch.nn.functional.softplus(std)
        return (mu.squeeze(-1), std.squeeze(-1))


class Critic(nn.Module):

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        v_min: float,
        v_max: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = GaussianQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            v_min=v_min,
            v_max=v_max,
            device=device,
        )
        self.qnet2 = GaussianQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            v_min=v_min,
            v_max=v_max,
            device=device,
        )
        self.device = device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (self.qnet1(obs, actions), self.qnet2(obs, actions))


class Actor(nn.Module):

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        std_min: float = -5,
        std_max: float = 0.5,
        log_std_init: float = --3,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.GELU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_act, device=device),
        )
        self.fc_log = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_act, device=device),
        )
        nn.init.uniform_(self.fc_mu[2].weight, -0.001, 0.001)
        nn.init.zeros_(self.fc_mu[2].bias)
        nn.init.uniform_(self.fc_log[2].weight, -0.001, 0.001)
        nn.init.constant_(self.fc_log[2].bias, log_std_init)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.register_buffer("mu_min", torch.as_tensor(-1, device=device))
        self.register_buffer("mu_max", torch.as_tensor(1, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(
        self, obs: torch.Tensor, is_deterministic=False, clip=None
    ) -> torch.Tensor:
        x = self.net(obs)
        mu = self.fc_mu(x)
        mu = torch.clamp(mu, self.mu_min, self.mu_max)
        log_std = self.fc_log(x)
        std = torch.clamp(log_std, self.std_min, self.std_max).exp()
        dist = GaussDistribution(mu, std, self.device)
        if clip is not None:
            if is_deterministic:
                return mu, None
            return dist.rsample_target(clip)
        if is_deterministic:
            actions = mu
            return actions, dist.log_prob(actions)
        return dist.rsample()

    def get_actions(self, obs, is_deterministic=False):
        x = self.net(obs)
        mu = self.fc_mu(x)
        mu = torch.clamp(mu, self.mu_min, self.mu_max)
        log_std = self.fc_log(x)
        std = torch.clamp(log_std, self.std_min, self.std_max).exp()
        dist = GaussDistribution(mu, std, self.device)
        if is_deterministic:
            return mu, None
        return dist.rsample()
