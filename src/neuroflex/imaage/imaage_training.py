import os
import time
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from neuroflex.utils.device_utils import get_available_device

from neuroflex.utils.printing_utils.printing_utils import Verbose
from neuroflex.utils.experiment_pipeline import Config

from neuroflex.utils.chatbot import load_original_model_for_causal_lm

from neuroflex.utils.rank_analysis.utils import extract_based_on_path


def get_AB_factorization(
        tensor: torch.Tensor,
        rank: int,
        trainable: list[bool],
        device: torch.device
) -> (torch.Tensor, torch.Tensor, float):
    """
    Computes the AB factorization of a given tensor.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        rank (int):
            The rank of the factorization.
        trainable (list[bool]):
            A list of booleans indicating whether the corresponding factor should be trainable.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The A factor of the factorization.
        (torch.Tensor):
            The B factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    out_shape, in_shape = tensor.shape

    # Initializing the AB factorization
    start_time = time.time()
    a = torch.randn(out_shape, rank).to(device)
    a.requires_grad = trainable[0]
    b = torch.randn(rank, in_shape).to(device)
    b.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return a, b, elapsed_time


def get_SVD_factorization(
        tensor: torch.Tensor,
        rank: int,
        trainable: list[bool],
        device: torch.device
) -> (torch.Tensor, torch.Tensor, torch.Tensor, float):
    """
    Computes the SVD factorization of a given tensor.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        rank (int):
            The rank of the factorization.
        trainable (list[bool]):
            A list of booleans indicating whether the corresponding factor should be trainable.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The U factor of the factorization.
        (torch.Tensor):
            The S factor of the factorization.
        (torch.Tensor):
            The V factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    # Initializing the SVD factorization
    start_time = time.time()
    u, s, vt = torch.svd(tensor.to("cpu"))
    us = torch.matmul(u[:, :rank], torch.diag(s[:rank])).to(device)
    us.requires_grad = trainable[0]
    vt = vt[:rank, :].to(device)
    vt.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return us, vt, elapsed_time


def imaage_training_trial(
        configuration: Config,
        verbose: Verbose = Verbose.SILENT
) -> None:
    """
    Compares which of the two initializations are better in terms of the quality loss and the speed of convergence to a
    good approximation.

    Args:
        configuration (Config):
            The configuration object.
        verbose (Verbose):
            The verbosity object.
    """

    device = get_available_device(
        preferred_device=configuration.get("device") if configuration.contains("device") else "cuda"
    )

    model_name = configuration.get("original_model_id").split("/")[-1]
    # Loading the model
    model = load_original_model_for_causal_lm(
        configuration,
        verbose=Verbose.INFO
    )

    # Extracting the candidate tensors for the analysis
    extracted_tensors = []
    extract_based_on_path(
        model,
        configuration.get("targets"),
        extracted_tensors,
        configuration.get("black_list"),
        verbose=verbose
    )
    # Choosing the actual tensors to analyze
    tensors_to_analyze = [extracted_tensors[0].get_tensor()]

    # Setting some parameters
    rank = configuration.get("rank")
    num_epochs = configuration.get("num_epochs") if configuration.contains("num_epochs") else 1000
    batch_size = configuration.get("batch_size") if configuration.contains("batch_size") else 1
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (16, 16)
    epoch_cut = configuration.get("epoch_cut") if configuration.contains("epoch_cut") else 750

    time_log = []
    csv_data = []
    for tensor_to_analyze in tensors_to_analyze:
        tensor_to_analyze = tensor_to_analyze.to(torch.float32)

        # Creating the figure to plot the results
        fig, axes = plt.subplots(2, 2, figsize=fig_size)

        # Preparing the data
        out_shape, in_shape = tensor_to_analyze.shape
        random_x = torch.randn(in_shape, batch_size).to(device)
        test_random_x = torch.randn(in_shape, batch_size).to(device)
        tensor_to_analyze = tensor_to_analyze.to(device)
        tensorx = torch.matmul(tensor_to_analyze, random_x)

        a, b, ab_time = get_AB_factorization(tensor_to_analyze, rank, [False, True], device)
        us, vt, svd_time = get_SVD_factorization(tensor_to_analyze, rank, [False, True], device)

        loss_types = ["activation loss", "tensor loss"]
        tensor_factrizations = {"A, B": [b, a], "U, S, V^T": [vt, us]}
        loss_histories_factorizations = {"activation loss": {}, "tensor loss": {}}

        if verbose >= Verbose.INFO:
            print()
        # Training the factorizations
        for factorization_label, factorization_init in tensor_factrizations.items():
            for loss_type in loss_types:
                # Cloning the tensors to avoid in-place operations
                factorization = [tensor.clone().detach() for tensor in factorization_init]
                for index in range(len(factorization)):
                    factorization[index].requires_grad = factorization_init[index].requires_grad

                # Storing the start time
                start_time = time.time()

                # Setting the optimizer
                trainable_tensors = [tensor for tensor in factorization if tensor.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable_tensors,
                    lr=configuration.get("learning_rate") if configuration.contains("learning_rate") else 1e-4,
                    eps=1e-7 if tensor_to_analyze.dtype == torch.float16 else 1e-8
                )

                activation_loss_history = []
                tensor_loss_history = []
                initial_activation_loss = None
                initial_tensor_loss = None

                if verbose >= Verbose.INFO:
                    print(f"Starting training using {loss_type} for {factorization_label}")
                for index in range(len(factorization)):
                    if verbose >= Verbose.INFO:
                        print(f"Tensor {index} in {factorization_label} requires grad: "
                              f"{factorization[index].requires_grad}")

                for epoch in tqdm(range(num_epochs)):
                    # Computing
                    x = random_x.clone().detach().to(device)
                    y = torch.eye(in_shape).to(device).to(device)
                    for tensor in factorization:
                        x = torch.matmul(tensor, x)
                        y = torch.matmul(tensor, y)

                    activation_loss = (torch.norm((tensorx - x), dim=0) ** 2).mean()
                    tensor_loss = torch.norm(tensor_to_analyze - y) ** 2

                    if initial_activation_loss is None:
                        initial_activation_loss = activation_loss.item()
                    if initial_tensor_loss is None:
                        initial_tensor_loss = tensor_loss.item()

                    if loss_type == "activation loss":
                        loss = activation_loss
                    elif loss_type == "tensor loss":
                        loss = tensor_loss
                    else:
                        raise ValueError(f"Unknown loss type: {loss_type}")

                    activation_loss_history.append(activation_loss.detach().cpu())
                    tensor_loss_history.append(tensor_loss.detach().cpu())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Storing the end time
                end_time = time.time()
                # Calculating and storing the time elapsed
                time_elapsed = end_time - start_time
                time_string = f"Time for training {factorization_label} using {loss_type}: {time_elapsed:.2f} seconds "\
                              f"{f'+ {svd_time:.2f} seconds to perform the SVD' if factorization_label == 'U, S, V^T' else ''}\n"
                time_log.append(time_string)
                if verbose >= Verbose.INFO:
                    print(time_string, flush=True)

                final_activation_loss = activation_loss_history[-1].item()
                final_tensor_loss = tensor_loss_history[-1].item()

                if verbose >= Verbose.INFO:
                    print("", flush=True)

                # Compute the test activation loss
                x_test = test_random_x.clone().detach()
                for tensor in factorization:
                    x_test = torch.matmul(tensor, x_test)
                test_activation_loss = (torch.norm((tensorx - x_test), dim=0) ** 2).mean().item()

                loss_histories_factorizations["activation loss"][
                    f"{factorization_label} trained using {loss_type}"] = activation_loss_history
                loss_histories_factorizations["tensor loss"][
                    f"{factorization_label} trained using {loss_type}"] = tensor_loss_history

                # Save details to CSV
                csv_data.append({
                    "Factorization": factorization_label,
                    "Loss Type": loss_type,
                    "Initial Activation Loss": initial_activation_loss,
                    "Final Activation Loss": final_activation_loss,
                    "Initial Tensor Loss": initial_tensor_loss,
                    "Final Tensor Loss": final_tensor_loss,
                    "Test Activation Loss": test_activation_loss,
                    "Training Time": time_elapsed,
                    "SVD Time": svd_time if factorization_label == "U, S, V^T" else 0.0
                })

        for label, activation_loss_history in loss_histories_factorizations["activation loss"].items():
            axes[0, 0].plot(activation_loss_history, label=f"{label}")
            axes[1, 0].plot(activation_loss_history[epoch_cut:], label=f"{label}")
            if configuration.contains("ylim"):
                axes[1, 0].set_ylim(0, configuration.get("ylim"))
            print(f"Activation loss for {label}: {activation_loss_history[-1]}")

        axes[0, 0].set_title("Full activation training loss history (target activation - approximated activation)")
        axes[1, 0].set_title(
            f"Loss history starting from epoch {epoch_cut} (target activation - approximated activation)")

        for label, tensor_loss_history in loss_histories_factorizations["tensor loss"].items():
            axes[0, 1].plot(tensor_loss_history, label=f"{label}", )
            axes[1, 1].plot(tensor_loss_history[epoch_cut:], label=f"{label}")
            if configuration.contains("ylim"):
                axes[1, 1].set_ylim(0, configuration.get("ylim"))
            print(f"Target tensor - approximated tensor for {label}: {tensor_loss_history[-1]}")

        axes[0, 1].set_title("Full loss history (target tensor - approximated tensor)")
        axes[1, 1].set_title(f"Loss history zoomed in from epoch {epoch_cut} (target tensor - approximated tensor)")

        for ax in axes.flatten():
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        output_path = os.path.join(configuration.get("path_to_storage"), model_name + "_plot.png") if configuration.contains(
            "path_to_storage") else "plot.png"
        plt.savefig(output_path)
        plt.show()

        for factorization_label, factorization in tensor_factrizations.items():
            for loss_type in loss_types:
                x_test = test_random_x.clone().detach()
                for tensor in factorization:
                    x_test = torch.matmul(tensor, x_test)
                activation_loss = (torch.norm((tensorx - x_test), dim=0) ** 2).mean()
                print(f"Activation loss for {factorization_label} using {loss_type} on test data: {activation_loss}")

    # Save the timing results to a file
    time_log_path = os.path.join(configuration.get("path_to_storage"), model_name + "_training_times.txt") if configuration.contains(
        "path_to_storage") else "training_times.txt"
    with open(time_log_path, "w") as f:
        f.writelines(time_log)

    # Save the loss results to a CSV file
    csv_path = os.path.join(configuration.get("path_to_storage"), model_name + "_losses.csv") if configuration.contains(
        "path_to_storage") else "losses.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["Factorization", "Loss Type", "Initial Activation Loss", "Final Activation Loss",
                      "Initial Tensor Loss", "Final Tensor Loss", "Test Activation Loss", "Training Time", "SVD Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


def imaage_train() -> None:
    """
    """

    pass
