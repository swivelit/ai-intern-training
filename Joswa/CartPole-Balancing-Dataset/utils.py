import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.savefig("plots/training_curve.png")
    plt.show()