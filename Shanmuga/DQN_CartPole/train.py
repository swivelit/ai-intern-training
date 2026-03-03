import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from model import DQN
from replay_buffer import ReplayBuffer

# ─── Hyperparameters ────────────────────────────────────────────────
EPISODES        = 800
GAMMA           = 0.99          # discount factor
LR              = 1e-3          # learning rate
BATCH_SIZE      = 64
BUFFER_CAPACITY = 10_000
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.995         # multiplicative decay per episode
TARGET_UPDATE   = 10            # sync target network every N episodes
SOLVE_REWARD    = 200           # CartPole-v1 is "solved" at avg 200
OUT_DIR         = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/DQN_CartPole"
# ────────────────────────────────────────────────────────────────────

def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = policy_net(torch.FloatTensor(state).unsqueeze(0))
        return q.argmax(dim=1).item()

def optimize(policy_net, target_net, optimizer, buffer):
    if len(buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    # Current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values (Bellman)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target_q = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    n_actions  = env.action_space.n               # 2

    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_CAPACITY)

    epsilon   = EPSILON_START
    ep_rewards = []
    solved_at  = None

    print("Training DQN on CartPole-v1 ...")
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = select_action(state, policy_net, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            optimize(policy_net, target_net, optimizer, buffer)

            state        = next_state
            total_reward += reward
            if done:
                break

        ep_rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Sync target network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Progress log
        if ep % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            print(f"  Episode {ep:>4}/{EPISODES}  |  "
                  f"Avg reward (last 50): {avg:6.1f}  |  "
                  f"eps = {epsilon:.3f}")

        # Check solved
        if len(ep_rewards) >= 100 and solved_at is None:
            if np.mean(ep_rewards[-100:]) >= SOLVE_REWARD:
                solved_at = ep
                print(f"\n  *** Solved at episode {ep} "
                      f"(avg reward >= {SOLVE_REWARD}) ***\n")

    env.close()
    return policy_net, ep_rewards, solved_at

def plot_rewards(ep_rewards, solved_at):
    window = 50
    rolling = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 5))
    plt.plot(ep_rewards, alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(range(window - 1, len(ep_rewards)),
             rolling, color="darkorange", linewidth=2,
             label=f"{window}-ep rolling avg")
    plt.axhline(SOLVE_REWARD, color="green", linestyle="--",
                linewidth=1.5, label=f"Solve threshold ({SOLVE_REWARD})")
    if solved_at:
        plt.axvline(solved_at, color="red", linestyle=":", linewidth=1.5,
                    label=f"Solved at ep {solved_at}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on CartPole-v1 - Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Reward curve saved: {path}")

def evaluate(policy_net, episodes=10):
    env   = gym.make("CartPole-v1")
    total = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            if terminated or truncated:
                break
        total.append(ep_r)
    env.close()
    avg = np.mean(total)
    print(f"Greedy evaluation ({episodes} eps): avg reward = {avg:.1f}")
    return avg

if __name__ == "__main__":
    policy_net, ep_rewards, solved_at = train()

    # Save weights
    weights_path = os.path.join(OUT_DIR, "dqn_cartpole.pth")
    torch.save(policy_net.state_dict(), weights_path)
    print(f"Model weights saved: {weights_path}")

    # Plot
    plot_rewards(ep_rewards, solved_at)

    # Final evaluation
    evaluate(policy_net)
