import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# ---------------------------
# Hyperparameters
# ---------------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100_000
EPISODES = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------
# DQN Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(DEVICE),
            torch.tensor(actions).to(DEVICE),
            torch.tensor(rewards).to(DEVICE),
            torch.tensor(next_states, dtype=torch.float32).to(DEVICE),
            torch.tensor(dones).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Training
# ---------------------------
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
rewards_history = []

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).to(DEVICE)
                action = policy_net(state_t).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = target_net(next_states).max(1)[0]
            target = rewards + GAMMA * next_q * (1 - dones.float())

            loss = nn.MSELoss()(q_values, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards_history.append(total_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    avg_reward = np.mean(rewards_history[-20:])
    print(f"Episode {episode:03d} | Reward: {total_reward:.1f} | Avg(20): {avg_reward:.1f}")

    if avg_reward >= 200:
        print("🎉 Solved CartPole!")
        break

# ---------------------------
# Save model
# ---------------------------
torch.save(policy_net.state_dict(), "models/dqn_cartpole.pth")

# ---------------------------
# Plot rewards
# ---------------------------
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN CartPole Training")
plt.savefig("plots/training_rewards.png")
plt.show()

env.close()
