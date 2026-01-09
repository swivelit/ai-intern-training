import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make("FrozenLake-v1", is_slippery=True)

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 5000
max_steps = 100

state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))
rewards = []

for ep in range(episodes):
    state, info = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Training finished")

with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("FrozenLake Q-Learning Reward Curve")
plt.savefig("reward_curve.png")
plt.show()
