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
To train the PPO agent:

```bash
python ppo.py
```

- Trains for 2048 timesteps for experimental purposes (can be increased for better performance)
- Updates the policy every 2048 steps (buffer size)
- Displays live progress with tqdm

## Features
- Continuous action sampling using MultivariateNormal
- State preprocessing (normalization + channel-first)
- On-policy buffer with advantage estimation
- Video logging and playback
- Modular class-based structure (ActorCritic, PPO, Memory)