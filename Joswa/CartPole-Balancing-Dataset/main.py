import gymnasium as gym
import torch
import torch.optim as optim
import random
import os

from dqn import DQN
from replay_buffer import ReplayBuffer
from utils import plot_rewards

# Create environment
env = gym.make("CartPole-v1")

# State and action sizes
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Models
model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)

target_model.load_state_dict(model.state_dict())

# Optimizer and replay buffer
optimizer = optim.Adam(model.parameters(), lr=0.001)
buffer = ReplayBuffer(10000)

# Hyperparameters
gamma = 0.99
batch_size = 64

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997

episodes = 600

rewards_list = []

# Select action
def select_action(state, epsilon):

    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return torch.argmax(q_values).item()


# Train function
def train():

    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    next_q = target_model(next_states).max(1)[0]

    target_q = rewards + gamma * next_q * (1 - dones)

    loss = torch.mean((current_q - target_q.detach()) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training loop
for episode in range(episodes):

    state, _ = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = select_action(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)

        train()

        state = next_state

        total_reward += reward

    rewards_list.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode + 1}, Reward: {total_reward}")


# Save model
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/dqn_cartpole.pth")

# Save plot
os.makedirs("plots", exist_ok=True)

plot_rewards(rewards_list)

env.close()