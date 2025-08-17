# PPO Agent for CarRacing-v3 (Gymnasium + PyTorch)

This project implements a **Proximal Policy Optimization (PPO)** reinforcement learning agent to play the `CarRacing-v3` environment from **Gymnasium** using **PyTorch**.

The agent learns to control a continuous-action car using policy gradient methods with clipped surrogate objective.

---

## Overview

- **Environment**: [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)
- **Algorithm**: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- **Deep Learning Framework**: PyTorch
- **Action Space**: Continuous `[steering, gas, brake]`
- **Training**: On-policy updates every 2048 steps

---

## Installation

Install the required packages:

```bash
pip install torch gymnasium[box2d,accept-rom-license] imageio tqdm numpy
```

Accept the license for Box2D when prompted by Gymnasium.

## Training

To train the PPO agent:

```bash
python ppo.py
```
- Trains for a configurable number of timesteps (default: 2048 for demo)
- On-policy updates every 2048 steps
- Automatically saves model checkpoints (.pth files)

## Features

- Proximal Policy Optimization (PPO) for continuous control
- On-policy buffer with advantage estimation
- Clipped surrogate objective for stable policy updates
- Grayscale image preprocessing
- MP4 video export of agent behavior
