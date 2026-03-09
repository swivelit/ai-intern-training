# DQN — CartPole Balancing

**Project 16** | Deep Reinforcement Learning | OpenAI Gymnasium

---

## What is CartPole?
CartPole is a classic RL control problem. A pole is attached to a cart moving on a frictionless track. The agent must keep the pole balanced upright by pushing the cart left or right.

- **State (4 values)**: Cart position, cart velocity, pole angle, pole angular velocity.
- **Actions (2)**: Push left or push right.
- **Reward**: +1 for every time step the pole stays upright.
- **Solved**: Average reward ≥ 200 over 100 consecutive episodes.

## Algorithm — Deep Q-Network (DQN)
DQN extends Q-Learning with two key innovations:
1. **Neural Network**: Approximates the Q-function instead of a look-up table.
2. **Experience Replay**: Stores transitions in a replay buffer and samples random mini-batches to break temporal correlations.
3. **Target Network**: A separate frozen copy of the network for stable Q-value targets.

| Hyperparameter | Value |
|---|---|
| Learning Rate | 0.001 |
| Gamma (discount) | 0.99 |
| Epsilon Decay | 0.995 per episode |
| Batch Size | 64 |
| Target Update Freq | Every 10 episodes |
| Replay Buffer | 10,000 transitions |

## Project Structure
| File | Description |
|---|---|
| `train.py` | Main training script |
| `model.py` | DQN neural network architecture |
| `replay_buffer.py` | Experience replay memory |
| `dqn_cartpole.pth` | Saved model weights |
| `reward_curve.png` | Training reward curve |

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Expected Output
- Console log showing reward improving every 50 episodes.
- Green dashed line on the curve at the 200-reward solve threshold.
- `dqn_cartpole.pth` — saved weights for the trained DQN agent.
- `reward_curve.png` — visual proof of learning progress.
