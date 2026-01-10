
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000
max_steps = 100

Q = np.zeros((state_size, action_size))
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    total_rewards = 0
    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
        total_rewards += reward
        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_rewards)

np.save("q_table.npy", Q)

plt.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode="valid"))
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("FrozenLake Q-Learning")
plt.show()
