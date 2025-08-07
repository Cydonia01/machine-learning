# A3C Agent for Atari Kung Fu Master (Gymnasium + PyTorch)

This project implements an **A3C (Advantage Actor-Critic)** reinforcement learning agent trained to play the Atari game **Kung Fu Master** using the Gymnasium library and PyTorch.

## Overview

- **Environment**: [KungFuMaster-v4](https://gymnasium.farama.org/environments/atari/kung_fu_master/)
- **RL Algorithm**: Advantage Actor-Critic (A3C variant)
- **Deep Learning Framework**: PyTorch
- **Environment Wrapper**: Frame stacking + grayscale + resizing for efficient learning
- **Training Setup**: Parallel training with 10 environments
- **Evaluation**: Average reward every 1000 steps
- **Video Output**: Saves an MP4 file showing the agent playing the game

---

## Requirements

Make sure to install the following Python packages:

```bash
pip install torch gymnasium[atari,accept-rom-license] ale-py imageio opencv-python tqdm
```

Ensure the ROM for Kung Fu Master is available (automatically installed via Gymnasium with license accepted).

## Training

You can train the agent by simply running the script:
```bash
python kungfu.py
```

- Uses 10 parallel environments for faster and more stable training.
- In the script, the trainining iterations is set to 4000 for experimental purposes. It can be adjusted based on your computational resources and desired training duration.

## Notes
- In this link you can find resources mentioned below: [Resources](https://drive.google.com/drive/folders/1xC2ZbnNIMv2HLNLVMLzr8XjBeFwp3OuF?usp=sharing)
- If you encounter issues with Gymnasium, you can manually download the ROM and place it in the appropriate directory.
- The final weights after lots of trainings are saved in `final_weights.pth`. You can load this model to evaluate the agent.
- You can find the sample video in the link above.