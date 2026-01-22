import gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

# -----------------------------
# Fix random seeds (optional)
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Hyperparameters
# -----------------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 10   # update target model every N episodes
EPISODES = 400

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Output folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr = LEARNING_RATE

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)

        states = np.zeros((BATCH_SIZE, self.state_size))
        next_states = np.zeros((BATCH_SIZE, self.state_size))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Predict Q-values
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        # Update Q-values using Bellman equation
        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target = rewards[i] + self.gamma * np.max(q_next[i])
            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        model_path = os.path.join(OUTPUT_DIR, "dqn_cartpole_model.keras")
        weights_path = os.path.join(OUTPUT_DIR, "dqn_cartpole_weights.weights.h5")

        self.model.save(model_path)
        self.model.save_weights(weights_path)

        print(f"\n✅ Model saved at: {model_path}")
        print(f"✅ Weights saved at: {weights_path}")


# -----------------------------
# Train DQN
# -----------------------------
def train_dqn():
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    rewards_history = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.replay()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-20:])
        avg_rewards.append(avg_reward)

        # Update target model
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        print(f"Episode: {episode}/{EPISODES} | Reward: {total_reward:.0f} | Avg(20): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        # Stop early if solved
        if avg_reward >= 200:
            print("\n🎉 Solved CartPole! Average reward ≥ 200")
            break

    env.close()

    # Save graph
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Reward per Episode")
    plt.plot(avg_rewards, label="Avg Reward (last 20)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training on CartPole-v1")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(OUTPUT_DIR, "training_rewards.png")
    plt.savefig(graph_path)
    plt.show()

    print(f"\n✅ Training graph saved at: {graph_path}")

    # Save model + weights
    agent.save()


if __name__ == "__main__":
    train_dqn()
