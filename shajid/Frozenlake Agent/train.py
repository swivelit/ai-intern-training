import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# Q-Learning Hyperparameters
# ----------------------------
ENV_NAME = "FrozenLake-v1"
IS_SLIPPERY = False   # False = easier (deterministic), True = harder (stochastic)
MAP_NAME = "4x4"

episodes = 8000
max_steps = 100

alpha = 0.8          # learning rate
gamma = 0.95         # discount factor

epsilon = 1.0        # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

# ----------------------------
# Create Environment
# ----------------------------
env = gym.make(ENV_NAME, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table
Q = np.zeros((n_states, n_actions))

# Store rewards
rewards_per_episode = []

# ----------------------------
# Training Loop
# ----------------------------
print("Training Q-Learning Agent...\n")

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_per_episode.append(total_reward)

    # Print progress
    if (ep + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-1000:])
        print(f"Episode: {ep+1}/{episodes} | Avg Reward (last 1000): {avg_reward:.3f} | Epsilon: {epsilon:.3f}")

print("\n✅ Training Completed!\n")

# ----------------------------
# Plot Reward Curve
# ----------------------------
window = 200
moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, label="Reward per Episode")
plt.plot(range(window-1, len(rewards_per_episode)), moving_avg, label=f"Moving Avg ({window})")
plt.title("FrozenLake Q-Learning Reward Curve")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Save Q-table
# ----------------------------
np.save("q_table_frozenlake.npy", Q)
print("✅ Q-table saved as: q_table_frozenlake.npy")

print("\nFinal Q-table:\n")
print(Q)

# ----------------------------
# Demonstrate Solved Environment
# ----------------------------
print("\n🎮 Demonstration of trained agent (5 episodes)...\n")

demo_env = gym.make(ENV_NAME, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode="human")

for demo_ep in range(5):
    state, _ = demo_env.reset()
    done = False
    total_reward = 0

    print(f"--- Demo Episode {demo_ep+1} ---")
    time.sleep(1)

    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, truncated, _ = demo_env.step(action)

        state = next_state
        total_reward += reward
        time.sleep(0.3)

        if done or truncated:
            break

    print(f"Reward: {total_reward}\n")
    time.sleep(1)

demo_env.close()
print("✅ Demo Completed!")
