import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN
from replay_buffer import ReplayBuffer


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 100

rewards = []

os.makedirs("saved", exist_ok=True)

def select_action(state, eps):
    if np.random.rand() < eps:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    return policy_net(state).argmax().item()

for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, EPSILON)
        next_state, reward, done, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) > BATCH_SIZE:
            s, a, r, ns, d = buffer.sample(BATCH_SIZE)
            s = torch.FloatTensor(s).to(device)
            a = torch.LongTensor(a).unsqueeze(1).to(device)
            r = torch.FloatTensor(r).to(device)
            ns = torch.FloatTensor(ns).to(device)
            d = torch.FloatTensor(d).to(device)

            q = policy_net(s).gather(1, a).squeeze()
            with torch.no_grad():
                max_next_q = target_net(ns).max(1)[0]
                target = r + GAMMA * max_next_q * (1 - d)

            loss = nn.MSELoss()(q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            break

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards.append(total_reward)

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    avg = np.mean(rewards[-50:])
    print(f"Episode {ep}, Reward: {total_reward}, Avg(50): {avg:.2f}")

    if avg >= 200:
        print("Environment solved!")
        break

torch.save(policy_net.state_dict(), "saved/dqn_cartpole.pth")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN CartPole Training")
plt.savefig("saved/training_plot.png")
plt.show()
