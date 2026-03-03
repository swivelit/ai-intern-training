import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# ─── Hyperparameters ────────────────────────────────────────────────
EPISODES        = 10_000    # total training episodes
LEARNING_RATE   = 0.8       # alpha
GAMMA           = 0.95      # discount factor
EPSILON_START   = 1.0       # start fully exploring
EPSILON_MIN     = 0.01      # minimum exploration
EPSILON_DECAY   = 0.0005    # decay per episode
EVAL_EVERY      = 1000      # print progress every N episodes
WINDOW          = 500       # rolling-average window for reward curve
# ────────────────────────────────────────────────────────────────────

def train():
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states  = env.observation_space.n   # 16
    n_actions = env.action_space.n        # 4

    # Initialise Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    epsilon   = EPSILON_START
    rewards   = []           # success flag per episode (1 = goal reached)

    print("Training Q-Learning agent on FrozenLake-v1 …")
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            # ε-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()          # explore
            else:
                action = np.argmax(Q[state, :])             # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellman update
            Q[state, action] += LEARNING_RATE * (
                reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]
            )

            state        = next_state
            total_reward += reward
            if done:
                break

        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        rewards.append(total_reward)

        if (ep + 1) % EVAL_EVERY == 0:
            win_rate = np.mean(rewards[-EVAL_EVERY:]) * 100
            print(f"  Episode {ep+1:>6}/{EPISODES}  |  "
                  f"Win rate (last {EVAL_EVERY}): {win_rate:.1f}%  |  "
                  f"eps = {epsilon:.4f}")

    env.close()
    return Q, rewards

def evaluate(Q, episodes=100):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(100):
            action = np.argmax(Q[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                wins += reward
                break
    env.close()
    return wins / episodes

def plot_rewards(rewards, save_dir):
    # Rolling average
    window = WINDOW
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.2, color="steelblue", label="Episode reward")
    plt.plot(range(window - 1, len(rewards)),
             rolling, color="darkorange", linewidth=2,
             label=f"{window}-episode rolling average")
    plt.xlabel("Episode")
    plt.ylabel("Reward  (1 = goal reached)")
    plt.title("FrozenLake-v1 – Q-Learning Reward Curve")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Reward curve saved: {path}")

def save_qtable(Q, save_dir):
    path = os.path.join(save_dir, "q_table.npy")
    np.save(path, Q)
    print(f"Q-table saved: {path}")

    # Pretty-print Q-table
    print("\nFinal Q-table (rows = states 0-15, cols = actions L/D/R/U):")
    print(np.round(Q, 3))

def demonstrate(Q, steps=20):
    env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=True)
    state, _ = env.reset()
    print("\n--- Demo run (greedy policy) ---")
    print(env.render())
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
    for step in range(steps):
        action = np.argmax(Q[state, :])
        state, reward, terminated, truncated, _ = env.step(action)
        print(f"Step {step+1}: {action_names[action]}")
        print(env.render())
        if terminated or truncated:
            if reward == 1.0:
                print("✅ Goal reached!")
            else:
                print("❌ Fell into a hole.")
            break
    env.close()

if __name__ == "__main__":
    OUT = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/FrozenLake_QLearning"

    Q, rewards = train()

    # Final evaluation
    win_rate = evaluate(Q)
    print(f"\nFinal evaluation win-rate over 100 episodes: {win_rate:.2%}")

    plot_rewards(rewards, OUT)
    save_qtable(Q, OUT)
    demonstrate(Q)
