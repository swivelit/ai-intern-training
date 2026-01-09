import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)

state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.001
min_epsilon = 0.01
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
        done = terminated or truncated

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    rewards.append(total_reward)

np.save("q_table.npy", q_table)

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("FrozenLake Q-Learning Reward Curve")
plt.savefig("reward_plot.png")
plt.show()

print("Training completed successfully!")
