import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# State & action size
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize Q-table
q_table = np.zeros((state_size, action_size))

# Hyperparameters
episodes = 3000
max_steps = 100

learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

# Rewards list
rewards = []

# ================= TRAINING =================
for episode in range(episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state
        total_rewards += reward

        if done:
            break

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    rewards.append(total_rewards)

print("✅ Training finished!\n")

# ================= SAVE OUTPUTS =================
np.save("outputs/q_table.npy", q_table)
np.save("outputs/rewards.npy", rewards)

print("📊 Q-Table:")
print(q_table)

# ================= PLOT =================
plt.plot(rewards)
plt.title("Reward Curve")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("outputs/reward_plot.png")
plt.show()

# ================= DEMO =================
print("\n🎮 Demonstration:\n")

state, _ = env.reset()

for step in range(50):
    print(f"Step {step + 1}")

    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if done:
        if reward == 1:
            print("✅ Agent reached goal!")
        else:
            print("❌ Agent fell into hole!")
        break

env.close()