import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dqn import DQN
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=50000)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
episodes = 400
target_update = 10

rewards_history = []

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state).argmax().item()

def train_step():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train_step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_history.append(total_reward)

    if ep % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), "models/dqn_cartpole.pth")

plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN CartPole Training")
plt.savefig("plots/rewards.png")
plt.show()
