import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

ENV_NAME = "FrozenLake-v1"
EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

env = gym.make(ENV_NAME, is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))
rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_table[state, action] += LEARNING_RATE * (
            reward + DISCOUNT * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

np.save("models/q_table.npy", q_table)

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("FrozenLake Q-Learning Reward Curve")
plt.savefig("plots/reward_curve.png")
plt.close()

print("✅ Training completed successfully!")
