# Deep Q-Learning (DQN) Agent for LunarLander-v3 (PyTorch + Gymnasium)

This project implements a **Deep Q-Network (DQN)** agent to solve the **LunarLander-v3** environment using **PyTorch** and **Gymnasium**.

---

## Overview

- **Environment**: [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- **Algorithm**: [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236)
- **Frameworks**: PyTorch, Gymnasium
- **Action Space**: Discrete (4 actions: do nothing, fire left, fire main, fire right)
- **Input**: 8-dimensional continuous state vector

---

## Installation

Install the required dependencies:

```bash
pip install torch gymnasium[box2d,accept-rom-license] imageio
```

Make sure to accept the Box2D license when prompted by Gymnasium.

---

## Training

To train the DQN agent:

```bash
python lunar_lander.py
```

- Trains for up to 2000 episodes (configurable)
- Uses experience replay and target network updates
- Saves a checkpoint if the environment is solved (average score ≥ 200)
- Optionally records training videos every 40 episodes

---

## Features

- Fully connected neural network for Q-value approximation
- Experience replay buffer for stable learning
- Epsilon-greedy action selection with decay
- Target network with soft updates
- Modular class-based structure (Network, Agent, ReplayMemory)
- Training and evaluation video logging
- Prints training progress and average scores

---

## Output

- Training progress and average scores per episode
- Saved model checkpoint (`checkpoint.pth`) when solved
- Training and evaluation videos (`training_video.mp4`, `video.mp4`)

---

## Note

- You can find the trained model checkpoint (`checkpoint.pth`), videos (`training_video.mp4`, `video.mp4`) in this [link](https://drive.google.com/drive/folders/1hahvTVjeFKXCmzTX-Hp58iV0I7ihplSH):

## Tips & Customization

- Adjust hyperparameters (learning rate, batch size, epsilon decay, etc.) for improved performance
- Increase the number of episodes for more robust training
- Use a GPU for faster training if available
- Visualize the agent's performance with the generated videos

---