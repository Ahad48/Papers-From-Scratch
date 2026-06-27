# Playing Atari with Deep Reinforcement Learning

- **Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller
- **Year:** 2013
- **Venue:** NIPS Deep Learning Workshop 2013
- **Link:** https://arxiv.org/abs/1312.5602
- **Key contributions:**
  - First deep learning model to learn control policies directly from raw pixel input via RL
  - Q-network maps observations → Q-values for each discrete action
  - Experience replay: random sampling from a fixed-size buffer breaks temporal correlations
  - Separate target network (frozen copy, updated every C steps) stabilises TD targets
  - Epsilon-greedy exploration annealed over training
- **Implementation scope:** Core DQN algorithm — QNetwork (MLP), ReplayBuffer, and DQNAgent with epsilon-greedy action selection and one-step Q-learning update. Targets standard Gym environments with flat observation spaces (e.g. CartPole-v1, LunarLander-v2). No Atari preprocessing or CNN. Unit tests verify shapes and forward pass only; no environment rollout.
