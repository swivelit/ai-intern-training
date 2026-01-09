import gymnasium as gym
import numpy as np
import pickle

# IMPORTANT: render_mode=None disables ALL rendering
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

state, info = env.reset()
done = False
step = 0

print("Starting demo run...\n")

while not done:
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {step}: State {state} -> Action {action} -> State {next_state}")
    state = next_state
    step += 1

print("\nEpisode finished with reward:", reward)
env.close()
