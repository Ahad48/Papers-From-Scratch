"""
Playing Atari with Deep Reinforcement Learning — Mnih et al., 2013
https://arxiv.org/abs/1312.5602

Targets flat-observation Gym environments (e.g. CartPole-v1, LunarLander-v2).
"""
import random
from collections import deque

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Section 4.1 — MLP approximating Q(s, a) for all actions simultaneously.

    Input:  (batch, obs_dim)
    Output: (batch, n_actions)  — one Q-value per discrete action
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        # TODO: Section 4.1 — two hidden layers, ReLU activations
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, obs_dim) → (batch, n_actions)
        return self.net(x)


class ReplayBuffer:
    """
    Section 4 — experience replay: stores (s, a, r, s', done) tuples.

    Random sampling breaks temporal correlation between consecutive transitions.
    """

    def __init__(self, capacity: int):
        # circular buffer; deque handles eviction automatically
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Returns (states, actions, rewards, next_states, dones) as tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states),
            torch.tensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Algorithm 1 — DQN agent with epsilon-greedy policy and target network.

    Maintains an online network (updated every step) and a target network
    (copied from online every target_update_freq steps) to stabilise TD targets.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10_000,  # steps over which epsilon anneals
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,  # steps between target network syncs
        hidden_dim: int = 128,
    ):
        self.online_net = QNetwork(obs_dim, n_actions, hidden_dim)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.step_counter = 0

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy action selection. Algorithm 1, line 7-8."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # random action
        else:
            with torch.no_grad():
                q_values = self.online_net(state.unsqueeze(0))  # add batch dimension
                return q_values.argmax().item()  # greedy action

    def update(self) -> float | None:
        """
        One gradient step on a sampled minibatch. Algorithm 1, lines 11-13.

        Returns the loss value, or None if buffer has fewer than batch_size transitions.

        TD target: r + gamma * max_a' Q_target(s', a')  (0 if done)
        """
        raise NotImplementedError

    def sync_target(self) -> None:
        """Copy online network weights to target network. Algorithm 1, line 10."""
        raise NotImplementedError
