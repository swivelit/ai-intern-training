import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

env = gym.make("FrozenLake-v1", is_slippery=True)

state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000
max_steps = 100

rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

np.save("outputs/q_table.npy", q_table)

plt.plot(np.convolve(rewards, np.ones(100)/100, mode="valid"))
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("FrozenLake Q-Learning")
plt.savefig("outputs/reward_curve.png")
plt.show()

print("Training completed.")
