# Deep Convolutional Q-Learning (DQN) Agent for MsPacman-v4 (PyTorch + Gymnasium)

This project implements a **Deep Q-Network (DQN)** agent using deep convolutional neural networks to solve the **MsPacman-v4** environment with **PyTorch** and **Gymnasium**.

---

## Overview

- **Environment**: [`MsPacman-v4`](https://gymnasium.farama.org/environments/atari/ms_pacman/)
- **Algorithm**: [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236)
- **Frameworks**: PyTorch, Gymnasium
- **Action Space**: Discrete (game actions)
- **Input**: Preprocessed RGB image, resized and normalized

---

## Installation

Make sure the required dependencies are installed:

```bash
pip install torch gymnasium[atari,accept-rom-license] imageio opencv-python pillow torchvision ale-py
```

Make sure to accept the Atari ROM license when prompted by Gymnasium.

---

## Training

To train the DQN agent:

```bash
python pacman.py
```

- Trains for up to 2000 episodes (can be adjusted)
- Uses experience replay and target network updates
- Saves a checkpoint if the environment is solved
- Optionally records training videos every 50 episodes

---

## Features

- Deep convolutional neural network for state representation
- Experience replay buffer
- Epsilon-greedy action selection with decay
- Target network for stable learning
- Frame preprocessing (resize, normalization)
- Video logging and playback
- Modular class-based structure (Network, Agent)

---

## Note

If you can't load the environment from Gymnasium, you can find the ROM file for Ms. Pac-Man [here](https://drive.google.com/drive/folders/1JF6UYXQxfHQK8Oy5b0ZRJ4CcgUWSvrxe)

## Output

- Training progress and average scores per episode
- Saved model checkpoint (`checkpoint.pth`) when solved
- Training and evaluation videos (`training_video.mp4`, `video.mp4`)

---

## References

- [DeepMind DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Gymnasium MsPacman Environment](https://gymnasium.farama.org/environments/atari/ms_pacman/)
