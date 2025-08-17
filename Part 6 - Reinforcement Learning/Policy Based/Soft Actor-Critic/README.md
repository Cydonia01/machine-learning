# Soft Actor-Critic (SAC) Agent for CarRacing-v3 (PyTorch + Gymnasium)

This project implements a **Soft Actor-Critic (SAC)** reinforcement learning agent to solve the **CarRacing-v3** continuous control environment using **PyTorch** and **Gymnasium**.

---

## Overview

- **Environment**: [`CarRacing-v3`](https://gymnasium.farama.org/environments/box2d/car_racing/)
- **Algorithm**: [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- **Frameworks**: PyTorch, Gymnasium
- **Action Space**: Continuous - `[steering, gas, brake]`
- **Input**: Preprocessed grayscale image, resized and flattened

---

## Installation

Make sure the required dependencies are installed:

```bash
pip install torch gymnasium[box2d,accept-rom-license] imageio opencv-python

```

Make sure to accept the Box2D license when prompted by Gymnasium.

## Training

To train the SAC agent:

```bash
python sac.py
```
- Trains the agent for a configurable number of episodes (default: 5 for demo/testing)
- Displays live progress with tqdm

## Features

- Continuous action sampling using MultivariateNormal
- State preprocessing (normalization + channel-first)
- Replay buffer for off-policy learning
- Dual Q-network for stability
- Target network soft updates
- Video logging and playback
- Modular class-based structure (Actor, Critic, Memory)
