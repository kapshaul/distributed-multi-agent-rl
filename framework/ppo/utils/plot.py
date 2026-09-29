import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_curves(train_reward_log, val_reward_log, train_step_log, train_steps):
    fig, ax1 = plt.subplots(figsize=(16, 9))

    color = 'tab:red'
    # Plot train & val loss
    ax1.plot(train_step_log, train_reward_log, c=color, alpha=0.25, label="Train Reward")
    ax1.plot(
        [np.ceil((i + 1) * train_steps) for i in range(len(val_reward_log))],
        val_reward_log,
        c="red",
        label="Evaluation Reward"
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_ylim(-0.01, 3)

    fig.tight_layout()
    ax1.legend(loc="center")
    plt.savefig("results/results.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_curves_train(train_reward_log, train_step_log):
    fig, ax1 = plt.subplots(figsize=(16, 9))

    window_size = 200
    rolling_avg = pd.Series(train_reward_log).rolling(window=window_size, min_periods=1).mean()

    color = 'tab:red'
    # Plot train & val loss
    ax1.plot(train_step_log, train_reward_log, c=color, alpha=0.25, label="Train Reward")
    ax1.plot(train_step_log, rolling_avg, c="red", label="Average Reward")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_ylim(-0.01, 3)

    fig.tight_layout()
    ax1.legend(loc="center")
    plt.savefig("results/results.png", dpi=300, bbox_inches="tight")
    plt.show()