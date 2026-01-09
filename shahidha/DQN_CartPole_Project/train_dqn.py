import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn_model import DQN
from replay_buffer import ReplayBuffer

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500
TARGET_UPDATE = 10
EPISODES = 500

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer()

epsilon = EPSILON_START
rewards_history = []

def select_action(state):
    global epsilon
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    return policy_net(state).argmax().item()

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + GAMMA * next_q_values * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)
    rewards_history.append(total_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    avg_reward = np.mean(rewards_history[-20:])
    print(f"Episode {episode}, Reward: {total_reward}, Avg(20): {avg_reward:.2f}")

    if avg_reward >= 200:
        print("Solved CartPole!")
        break

torch.save(policy_net.state_dict(), "dqn_cartpole.pth")

plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Reward")
plt.savefig("rewards.png")
plt.show()
