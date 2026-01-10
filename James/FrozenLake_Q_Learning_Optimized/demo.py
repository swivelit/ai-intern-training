import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=True)

q_table = np.load("models/q_table.npy")

state, _ = env.reset()
done = False

print(env.render())

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(env.render())

env.close()
