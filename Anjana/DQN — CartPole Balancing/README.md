
# DQN CartPole-v1

Deep Q-Network (DQN) implemented **from scratch** using PyTorch.

## Features
- Experience Replay
- Target Network
- ε-greedy exploration
- Solves CartPole (avg reward ≥200)

## Files
- dqn.py        : DQN neural network
- train.py      : training loop (OpenAI Gym)
- plot.png      : training rewards
- weights.pth   : trained weights (placeholder)

## Run
```bash
pip install gym torch matplotlib
python train.py
```
