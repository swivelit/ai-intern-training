import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", render_mode="human")
q_table = np.load("outputs/q_table.npy")

for episode in range(5):
    state, _ = env.reset()
    done = False
    print(f"\nEpisode {episode + 1}")

    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.5)

        if terminated or truncated:
            print("✅ Success!" if reward == 1 else "❌ Failed")
            done = True

env.close()
