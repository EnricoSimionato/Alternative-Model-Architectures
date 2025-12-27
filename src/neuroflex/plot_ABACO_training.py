import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    loss_file_path = "gemma-2-2b_abaco_experiment_version_10_training_log_loss.json"
    val_loss_file_path = "gemma-2-2b_abaco_experiment_version_10_training_log_val_loss.json"
    exp_file_path = "gemma-2-2b_abaco_experiment_version_10_training_log_alpha.json"

    with open(loss_file_path, "r") as file:
        data = json.load(file)
    timestamps = np.array([entry[0] for entry in data])
    steps = np.array([entry[1] for entry in data])
    losses = np.array([entry[2] for entry in data])

    # Load validation loss data
    with open(val_loss_file_path, "r") as file:
        val_data = json.load(file)
    val_steps = np.array([entry[1] for entry in val_data])
    val_losses = np.array([entry[2] for entry in val_data])

    with open(exp_file_path, "r") as file:
        data = json.load(file)
    alphas = np.array([entry[2] for entry in data])

    # Normalize timestamps to start from zero
    timestamps -= timestamps[0]

    # Define a function for smoothing
    def smooth_curve(values, window_size=10):
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    # Apply smoothing to the loss values
    window_size = 10
    # Adjust for more or less smoothing
    smoothed_losses = smooth_curve(losses, window_size)
    smoothed_steps = steps[:len(smoothed_losses)]  # Align steps with smoothed losses

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot smoothed training loss on the primary y-axis
    train_loss_line, = ax1.plot(smoothed_steps, smoothed_losses, label="Smoothed Training Loss", color="#405292", linewidth=1.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss value")

    val_loss_line, = ax1.plot(val_steps, val_losses, label="Validation Loss", color="#e89b4b", linewidth=1.5)

    # Create a secondary y-axis for the exponential function
    ax2 = ax1.twinx()
    alpha_line, = ax2.plot(steps, alphas, label=f"Alpha exponential decay", color="#D54040", linestyle="dashed", linewidth=1.0)
    ax2.set_ylabel("Alpha")
    ax2.tick_params(axis='y', labelcolor="#D54040")

    # Improve grid visibility with solid lines
    ax1.grid(True, linestyle='-', linewidth=0.5)

    all_lines = [train_loss_line, val_loss_line, alpha_line]  # Combine handles from both axes
    all_labels = [l.get_label() for l in all_lines]

    ax1.legend(all_lines, all_labels, loc="upper left", fontsize=10, frameon=True)  # Unified legend
    #ax1.legend(all_lines, all_labels, loc="upper left", fontsize=10, frameon=True, bbox_to_anchor=(0.1, 1))


    plt.tight_layout()

    # Title and legend
    #fig.suptitle("Smoothed Training Loss History with Exponential Decay")

    # Show plot
    name = "training_loss_plot.pdf"
    plt.savefig(name, format="pdf")

if __name__ == "__main__":
    main()