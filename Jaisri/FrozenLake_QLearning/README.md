# FrozenLake Agent — Q-Learning (Reinforcement Learning)

**Project 15** | Reinforcement Learning | OpenAI Gymnasium

---

## What is FrozenLake?
FrozenLake is a grid-world environment where an agent must navigate from **Start (S)** to **Goal (G)** across a frozen lake, avoiding **Holes (H)**. The lake is slippery, so the agent doesn't always move in the intended direction.

```
S F F F
F H F H
F F F H
H F F G
```

## Algorithm — Tabular Q-Learning
Q-Learning is a model-free Reinforcement Learning algorithm. It learns a **Q-table** mapping every `(state, action)` pair to an expected future reward using the **Bellman equation**:

```
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') − Q(s, a)]
```

| Symbol | Meaning |
|--------|---------|
| α (alpha) | Learning rate = 0.8 |
| γ (gamma) | Discount factor = 0.95 |
| ε (epsilon) | Exploration rate (decays 1.0 → 0.01) |

## Project Structure
| File | Description |
|------|-------------|
| `train.py` | Main script — trains, evaluates, plots, and demonstrates the agent |
| `q_table.npy` | Saved Q-table (16×4 numpy array) |
| `reward_curve.png` | Training reward curve with rolling average |
| `requirements.txt` | Python dependencies |

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

## Expected Output
- Console log of win-rate every 1,000 episodes.
- Final evaluation win-rate over 100 greedy episodes.
- `reward_curve.png` — shows the agent learning over time.
- `q_table.npy` — the learned Q-table ready for deployment.
- A step-by-step demonstration of the greedy policy in the terminal.
