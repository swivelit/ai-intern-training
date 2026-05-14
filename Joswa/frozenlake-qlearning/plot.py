import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("rewards.npy")

plt.plot(rewards)
plt.title("Training Rewards")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()