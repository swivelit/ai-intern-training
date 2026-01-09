import gymnasium as gym
import numpy as np
import time

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
q_table = np.load("q_table.npy")

state, _ = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.5)

if reward == 1:
    print("🎉 Agent reached the goal!")
else:
    print("❌ Agent failed.")
