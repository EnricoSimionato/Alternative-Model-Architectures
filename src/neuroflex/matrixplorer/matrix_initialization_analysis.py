import os
import time
import csv
from tqdm import tqdm
import pickle as pkl

import matplotlib.pyplot as plt

import torch

from neuroflex.utils.device_utils import get_available_device

from neuroflex.utils.printing_utils.printing_utils import Verbose
from neuroflex.utils.experiment_pipeline import Config

from neuroflex.utils.chatbot import load_original_model_for_causal_lm

from neuroflex.utils.rank_analysis.utils import extract_based_on_path

from neuroflex.utils.rank_analysis.utils import AnalysisTensorDict


def get_ab_factorization(
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
    a = torch.randn(out_shape, rank, dtype=tensor.dtype).to(device)
    a.requires_grad = trainable[0]
    b = torch.randn(rank, in_shape, dtype=tensor.dtype).to(device)
    b.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return a, b, elapsed_time


def get_svd_factorization(
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
    u, s, v = torch.svd(tensor.to(torch.float32).to("cpu"))
    us = torch.matmul(u[:, :rank], torch.diag(s[:rank])).to(tensor.dtype).to(device)
    us.requires_grad = trainable[0]
    vt = v.T
    vt = vt[:rank, :].to(tensor.dtype).to(device)
    vt.requires_grad = trainable[1]
    elapsed_time = time.time() - start_time

    return us, vt, elapsed_time


def get_global_matrix_factorization(
        tensor: torch.Tensor,
        global_matrix: torch.Tensor,
        rank: int,
        trainable: bool,
        initialization_type: str,
        device: torch.device
) -> (torch.Tensor, torch.Tensor, float):
    """
    Computes the factorization of a given tensor using a global matrix.

    Args:
        tensor (torch.Tensor):
            The tensor to be factorized.
        global_matrix (torch.Tensor):
            The global matrix to be used in the factorization.
        rank (int):
            The rank of the factorization.
        trainable (bool):
            A boolean indicating whether the factor should be trainable.
        initialization_type (str):
            The type of initialization to use.
        device (torch.device):
            The device to perform the factorization on.

    Returns:
        (torch.Tensor):
            The factor of the factorization.
        (torch.Tensor):
            The factor of the factorization.
        (float):
            The time elapsed to compute the factorization.
    """

    out_shape, in_shape = tensor.shape

    if global_matrix.shape[0] != out_shape or global_matrix.shape[1] != rank:
        raise ValueError("The global matrix must have the shape (out_shape, rank).")

    start_time = time.time()
    if initialization_type == "random":
        b = torch.randn(rank, in_shape, dtype=tensor.dtype).to(device)
        b.requires_grad = trainable

    elif initialization_type == "pseudo-inverse":
        b = torch.matmul(
            torch.linalg.pinv(global_matrix.to(torch.float32).to("cpu")).to(device),
            tensor.to(torch.float32).to(device)
        ).to(tensor.dtype)
        b.requires_grad = trainable

    else:
        raise ValueError("Unknown initialization type.")
    elapsed_time = time.time() - start_time

    return global_matrix, b, elapsed_time


def perform_simple_initialization_analysis(
        configuration: Config,
) -> None:
    """
    Compares which of the two initializations are better in terms of the quality loss and the speed of convergence to a
    good approximation.

    Args:
        configuration (Config):
            The configuration object.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name = configuration.get("file_name")
    file_name_no_format = file_name.split(".")[0]

    # Getting the parameters related to the analysis from the configuration
    verbose = configuration.get_verbose()
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)
    rank = configuration.get("rank")
    num_epochs = configuration.get("num_epochs") if configuration.contains("num_epochs") else 5000
    batch_size = configuration.get("batch_size") if configuration.contains("batch_size") else 64
    epoch_cut = configuration.get("epoch_cut") if configuration.contains("epoch_cut") else 750
    device = get_available_device(preferred_device=configuration.get("device") if configuration.contains("device") else "cuda")

    if file_available:
        print(f"The file '{file_path}' is available.")
        # Load the data from the file
        with open(file_path, "rb") as f:
            all_loss_histories = pkl.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(configuration)

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
        tensor_wrappers_to_analyze = extracted_tensors

        time_log = []
        csv_data = []
        all_loss_histories = {}
        for tensor_wrapper_to_analyze in tensor_wrappers_to_analyze:
            verbose.print(f"\nAnalyzing tensor: {tensor_wrapper_to_analyze.get_path()}", Verbose.INFO)
            tensor_to_analyze = tensor_wrapper_to_analyze.get_tensor().to(torch.float32)

            # Preparing the data
            out_shape, in_shape = tensor_to_analyze.shape
            random_x = torch.randn(in_shape, batch_size).to(device)
            test_random_x = torch.randn(in_shape, batch_size).to(device)
            tensor_to_analyze = tensor_to_analyze.to(device)
            tensorx = torch.matmul(tensor_to_analyze, random_x)

            # Initializing the factorizations
            a, b, ab_time = get_ab_factorization(tensor_to_analyze, rank, [False, True], device)
            us, vt, svd_time = get_svd_factorization(tensor_to_analyze, rank, [False, True], device)

            tensor_factorizations = {
                "U, S, V^T": {"tensors": [vt, us], "init_time": svd_time},
                "A, B": {"tensors": [b, a], "init_time": ab_time}
            }

            loss_types = ["activation loss", "tensor loss"]
            loss_histories_factorizations = {"activation loss": {}, "tensor loss": {}}

            # Training the factorizations
            for factorization_label, factorization_init_and_init_time in tensor_factorizations.items():
                factorization_init = factorization_init_and_init_time["tensors"]
                init_time = factorization_init_and_init_time["init_time"]
                for loss_type in loss_types:
                    # Cloning the tensors to avoid in-place operations
                    factorization = [tensor.clone().detach() for tensor in factorization_init]
                    for index in range(len(factorization)):
                        factorization[index].requires_grad = factorization_init[index].requires_grad

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

                    verbose.print(f"\nStarting training using {loss_type} for {factorization_label}", Verbose.INFO)
                    for index in range(len(factorization)):
                        verbose.print(
                            f"Tensor {index} in {factorization_label} requires grad: "
                            f"{factorization[index].requires_grad}", Verbose.INFO
                        )

                    start_time = time.time()
                    # Training the factorization
                    for _ in tqdm(range(num_epochs)):
                        x = random_x.clone().detach().to(device)
                        y = torch.eye(in_shape).to(device)
                        for tensor in factorization:
                            x = torch.matmul(tensor, x)
                            y = torch.matmul(tensor, y)

                        # Computing the losses
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

                        # Storing the losses
                        activation_loss_history.append(activation_loss.detach().cpu())
                        tensor_loss_history.append(tensor_loss.detach().cpu())

                        # Performing the optimization step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    end_time = time.time()
                    # Calculating and storing the time elapsed
                    time_elapsed = end_time - start_time
                    time_string = (f"Time for training {factorization_label} using {loss_type}: {time_elapsed:.2f} "
                                   f"seconds + {init_time:.2f} seconds to initialize the matrix\n")
                    time_log.append(time_string)
                    verbose.print(time_string, Verbose.INFO)

                    final_activation_loss = activation_loss_history[-1].item()
                    final_tensor_loss = tensor_loss_history[-1].item()
                    verbose.print(
                        f"Final activation loss for {factorization_label} trained using {loss_type}: "
                        f"{final_activation_loss}",
                        Verbose.INFO
                    )
                    verbose.print(
                        f"Final tensor loss for {factorization_label} trained using {loss_type}: "
                        f"{final_tensor_loss}",
                        Verbose.INFO
                    )

                    # Computing the test activation loss
                    x_test = test_random_x.clone().detach()
                    for tensor in factorization:
                        x_test = torch.matmul(tensor, x_test)
                    test_activation_loss = (torch.norm((torch.matmul(tensor_to_analyze, test_random_x) - x_test), dim=0) ** 2).mean().item()
                    verbose.print(
                        f"Test activation loss for {factorization_label} trained using {loss_type}: "
                        f"{test_activation_loss}",
                        Verbose.INFO
                    )

                    loss_histories_factorizations["activation loss"][
                        f"{factorization_label} trained using {loss_type}"] = activation_loss_history
                    loss_histories_factorizations["tensor loss"][
                        f"{factorization_label} trained using {loss_type}"] = tensor_loss_history

                    # Save details to CSV
                    csv_data.append({
                        "Original Tensor Path": tensor_wrapper_to_analyze.get_path(),
                        "Original Tensor Shape": tensor_wrapper_to_analyze.get_shape(),
                        "Factorization": factorization_label,
                        "Loss Type": loss_type,
                        "Initial Activation Loss": initial_activation_loss,
                        "Final Activation Loss": final_activation_loss,
                        "Initial Tensor Loss": initial_tensor_loss,
                        "Final Tensor Loss": final_tensor_loss,
                        "Test Activation Loss": test_activation_loss,
                        "Training Time": time_elapsed,
                        "Initialization Time": init_time
                    })

            all_loss_histories[tensor_wrapper_to_analyze.get_path()] = loss_histories_factorizations

        # Saving the loss histories to a file
        with open(file_path, "wb") as f:
            pkl.dump(all_loss_histories, f)

        # Saving the timing results to a file
        time_log_path = os.path.join(directory_path, file_name_no_format + "_training_times.txt")
        with open(time_log_path, "w") as f:
            f.writelines(time_log)

        # Saving the loss results to a CSV file
        csv_path = os.path.join(directory_path, file_name_no_format + f"_losses_and_times.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Original Tensor Path", "Original Tensor Shape", "Factorization", "Loss Type",
                          "Initial Activation Loss", "Final Activation Loss", "Initial Tensor Loss",
                          "Final Tensor Loss", "Test Activation Loss", "Training Time", "Initialization Time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

    for tensor_to_analyze_label, loss_histories_factorizations in all_loss_histories.items():
        # Creating the figure to plot the results
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle(f"Initialization analysis of the tensor: {tensor_to_analyze_label}")
        for label, activation_loss_history in loss_histories_factorizations["activation loss"].items():
            axes[0, 0].plot(activation_loss_history, label=f"{label}")
            axes[1, 0].plot(activation_loss_history, label=f"{label}")
            if configuration.contains("y_range"):
                axes[1, 0].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 0].set_xlim(configuration.get("x_range"))
            print(f"Activation loss for {label}: {activation_loss_history[-1]}")

        axes[0, 0].set_title("Full activation training loss history (target activation - approximated activation)")
        axes[1, 0].set_title(
            f"Loss history starting from epoch {epoch_cut} (target activation - approximated activation)")

        for label, tensor_loss_history in loss_histories_factorizations["tensor loss"].items():
            axes[0, 1].plot(tensor_loss_history, label=f"{label}", )
            axes[1, 1].plot(tensor_loss_history, label=f"{label}")
            if configuration.contains("y_range"):
                axes[1, 1].set_ylim(configuration.get("y_range"))
            if configuration.contains("x_range"):
                axes[1, 1].set_xlim(configuration.get("x_range"))
            print(f"Target tensor - approximated tensor for {label}: {tensor_loss_history[-1]}")

        axes[0, 1].set_title("Full loss history (target tensor - approximated tensor)")
        axes[1, 1].set_title(f"Loss history zoomed in from epoch {epoch_cut} (target tensor - approximated tensor)")

        for ax in axes.flatten():
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()

        # Saving the plot
        plt.savefig(os.path.join(directory_path, file_name_no_format + f"_{tensor_to_analyze_label}_plot.png"))


def perform_global_matrices_initialization_analysis(
        configuration: Config,
) -> None:
    """
    Compares which of the two initializations are better in terms of the quality loss and the speed of convergence to a
    good approximation for the global matrices' framework.

    Args:
        configuration (Config):
            The configuration object.
    """

    # Setting some parameters
    rank = configuration.get("rank")
    num_epochs = configuration.get("num_epochs") if configuration.contains("num_epochs") else 1000
    batch_size = configuration.get("batch_size") if configuration.contains("batch_size") else 1
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (16, 16)
    epoch_cut = configuration.get("epoch_cut") if configuration.contains("epoch_cut") else 750
    device = get_available_device(
        preferred_device=configuration.get("device") if configuration.contains("device") else "cuda"
    )
    verbose = configuration.get_verbose()

    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name = configuration.get("file_name")
    file_name_no_format = file_name.split(".")[0]

    # Creating the figure to plot the results
    fig, axes = plt.subplots(2, 3, figsize=fig_size)

    if file_available:
        print(f"The file '{file_path}' is available.")
        # Load the data from the file
        with open(file_path, "rb") as f:
            tensor_factorizations_loss_histories, tensor_factorizations_dict = pkl.load(f)
    else:
        # Loading the model
        model = load_original_model_for_causal_lm(configuration)

        # Extracting the candidate tensors for the analysis
        extracted_tensor_wrappers = []
        extract_based_on_path(
            model,
            configuration.get("targets"),
            extracted_tensor_wrappers,
            configuration.get("black_list"),
            verbose=verbose
        )

        # Choosing the actual tensors to analyze
        tensor_wrappers_key_for_analysis = extracted_tensor_wrappers[0].get_label()
        tensor_wrappers_to_analyze = [
            tensor_wrapper
            for tensor_wrapper in extracted_tensor_wrappers
            if tensor_wrapper.get_label() == tensor_wrappers_key_for_analysis
        ]
        # Defining the shape of the analyzed matrices
        shape = tensor_wrappers_to_analyze[0].get_shape()
        # Checking the shapes of the tensors are all the same
        for tensor in tensor_wrappers_to_analyze:
            if tensor.get_shape() != shape:
                raise ValueError("The tensors to analyze must have the same shape.")
        verbose.print(f"\nShape of the tensors to analyze: {shape}", Verbose.INFO)

        # Defining the tensor dictionaries to compare different initializations
        tensors_to_analyze_ab = AnalysisTensorDict(
            [tensor_wrappers_key_for_analysis] * len(tensor_wrappers_to_analyze), tensor_wrappers_to_analyze
        )
        tensors_to_analyze_pseudo_inverse = AnalysisTensorDict(
            [tensor_wrappers_key_for_analysis]*len(tensor_wrappers_to_analyze), tensor_wrappers_to_analyze
        )
        tensors_to_analyze_svd = AnalysisTensorDict(
            [tensor_wrappers_key_for_analysis]*len(tensor_wrappers_to_analyze), tensor_wrappers_to_analyze
        )
        tensors_to_analyze_ab.set_dtype(torch.float32)
        tensors_to_analyze_pseudo_inverse.set_dtype(torch.float32)
        tensors_to_analyze_svd.set_dtype(torch.float32)

        # Defining the global matrix to use in the analysis
        global_matrix = torch.randn(shape[0], rank).to(device)
        global_matrix.requires_grad = False

        # Initializing the factorizations using random initialization
        random_init_time = 0.0
        for tensor in tensors_to_analyze_ab.get_tensor_list(tensor_wrappers_key_for_analysis):
            tensor.set_attribute("factorization_type", "AB randomly initialized")
            tensor.set_attribute("global_matrix", global_matrix)
            a, b, random_init_time_one_matrix = get_global_matrix_factorization(
                tensor.get_tensor(),
                global_matrix,
                rank,
                True,
                "random",
                device
            )
            random_init_time += random_init_time_one_matrix
            tensor.set_attribute("factorization", [b, a])

        # Initializing the factorizations using pseudo-inverse initialization
        preudo_inverse_init_time = 0.0
        for tensor in tensors_to_analyze_pseudo_inverse.get_tensor_list(tensor_wrappers_key_for_analysis):
            tensor.set_attribute("factorization_type", "AB pseudo-inverse initialized")
            tensor.set_attribute("global_matrix", global_matrix)
            a, b, random_init_time_one_matrix = get_global_matrix_factorization(
                tensor.get_tensor(),
                global_matrix,
                rank,
                True,
                "pseudo-inverse",
                device
            )
            preudo_inverse_init_time += random_init_time_one_matrix
            tensor.set_attribute("factorization", [b, a])

        # Initializing the factorizations using SVD initialization
        svd_init_time = 0.0
        for tensor in tensors_to_analyze_svd.get_tensor_list(tensor_wrappers_key_for_analysis):
            tensor.set_attribute("factorization_type", "AB pseudo-inverse initialized")
            tensor.set_attribute("global_matrix", global_matrix)
            us, vt, svd_init_time_one_matrix = get_svd_factorization(
                tensor.get_tensor(),
                rank,
                [True, True],
                device
            )
            svd_init_time += svd_init_time_one_matrix
            tensor.set_attribute("factorization", [vt, us])

        if verbose >= Verbose.INFO:
            print(f"\nInitialization times:\n"
                  f"\tAB random initialization: {random_init_time:.2f} seconds, "
                  f"\tAB pseudo-inverse initialization: {preudo_inverse_init_time:.2f} seconds, "
                  f"\tSVD initialization: {svd_init_time:.2f} seconds")

        tensor_factorizations_dict = {
            "AB randomly initialized": [tensors_to_analyze_ab, random_init_time],
            "AB pseudo-inverse initialized": [tensors_to_analyze_pseudo_inverse, preudo_inverse_init_time],
            "SVD initialized": [tensors_to_analyze_svd, svd_init_time]
        }
        tensor_factorizations_losses = ["tensor loss", "tensor loss", "penalized tensor loss"]

        time_log = []
        csv_data = []
        tensor_factorizations_loss_histories = {"activation loss": {}, "tensor loss": {}, "penalization term": {}}

        if verbose >= Verbose.INFO:
            print()
        for tensor_factorization_index, tensor_factorization_key_value in enumerate(tensor_factorizations_dict.items()):
            tensor_factorization_key, tensor_factorization_value = tensor_factorization_key_value
            factorization_label = tensor_factorization_key
            tensors_to_analyze_dict = tensor_factorization_value[0]
            tensor_init_time = tensor_factorization_value[1]

            # Preparing the data
            wrapper_tensors_to_analyze = tensors_to_analyze_dict.get_tensor_list(tensors_to_analyze_dict.get_keys()[0])
            out_shape, in_shape = wrapper_tensors_to_analyze[0].get_shape()
            random_x = torch.randn(in_shape, batch_size).to(device)
            test_random_x = torch.randn(in_shape, batch_size).to(device)
            tensors_to_analyze = [wrapper_tensor.get_tensor().to(device) for wrapper_tensor in wrapper_tensors_to_analyze]
            tensorsx = [torch.matmul(tensor, random_x) for tensor in tensors_to_analyze]

            # Setting the optimizer
            trainable_tensors = [tensor for tensor_wrapper in wrapper_tensors_to_analyze for tensor in tensor_wrapper.get_attribute("factorization") if tensor.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_tensors,
                lr=configuration.get("learning_rate") if configuration.contains("learning_rate") else 1e-4,
                eps=1e-7 if trainable_tensors[0].dtype == torch.float16 else 1e-8
            )

            activation_loss_history = []
            tensor_loss_history = []
            penalization_term_history = []
            initial_activation_loss = None
            initial_tensor_loss = None
            initial_penalization_term = None

            if verbose >= Verbose.INFO:
                print(f"Starting training using tensor loss for {factorization_label}")

            # Storing the start time
            start_time = time.time()
            for _ in tqdm(range(num_epochs)):
                total_activation_loss = torch.Tensor([0.0]).to(device)
                total_tensor_loss = torch.Tensor([0.0]).to(device)

                for index, tensor_wrapper in enumerate(wrapper_tensors_to_analyze):
                    x = random_x.clone().detach().to(device)
                    y = torch.eye(in_shape).to(device).to(device)
                    for factorization_term in tensor_wrapper.get_attribute("factorization"):
                        x = torch.matmul(factorization_term, x)
                        y = torch.matmul(factorization_term, y)

                    total_activation_loss += (torch.norm((tensorsx[index] - x), dim=0) ** 2).mean()
                    total_tensor_loss += torch.norm(tensors_to_analyze[index] - y) ** 2

                if initial_activation_loss is None:
                    initial_activation_loss = total_activation_loss.item()
                if initial_tensor_loss is None:
                    initial_tensor_loss = total_tensor_loss.item()

                activation_loss_history.append(total_activation_loss.detach().cpu())
                tensor_loss_history.append(total_tensor_loss.detach().cpu())

                if tensor_factorizations_losses[tensor_factorization_index] == "activation loss":
                    loss = total_activation_loss
                elif tensor_factorizations_losses[tensor_factorization_index] == "tensor loss":
                    loss = total_tensor_loss
                elif tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss":
                    penalization_term = torch.Tensor([0.0]).to(device)
                    for index_1, tensor_wrapper_1 in enumerate(wrapper_tensors_to_analyze):
                        for index_2, tensor_wrapper_2 in enumerate(wrapper_tensors_to_analyze):
                            if index_2 > index_1:
                                penalization_term += 2.5 * torch.norm((tensor_wrapper_1.get_attribute("factorization")[1].to(device) - tensor_wrapper_2.get_attribute("factorization")[1].to(device)) ** 2)
                    loss = total_tensor_loss + penalization_term

                    if initial_penalization_term is None:
                        initial_penalization_term = penalization_term

                    penalization_term_history.append(penalization_term.detach().cpu())
                else:
                    raise ValueError(f"Unknown loss type: {tensor_factorizations_losses[tensor_factorization_index]}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Storing the end time
            end_time = time.time()
            # Calculating and storing the time elapsed
            time_elapsed = end_time - start_time

            time_string = f"Time for training {factorization_label} using tensor loss: {time_elapsed:.2f} seconds + {tensor_init_time:.2f} seconds to initialize the factorization\n"
            time_log.append(time_string)

            if verbose >= Verbose.INFO:
                print(time_string, flush=True)

            final_activation_loss = activation_loss_history[-1].item()
            final_tensor_loss = tensor_loss_history[-1].item()

            if verbose >= Verbose.INFO:
                print("", flush=True)

            # Compute the test activation loss
            total_test_activation_loss = torch.Tensor([0.0]).to(device)
            for index, tensor_wrapper in enumerate(wrapper_tensors_to_analyze):
                x_test = random_x.clone().detach().to(device)
                for factorization_term in tensor_wrapper.get_attribute("factorization"):
                    x_test = torch.matmul(factorization_term.to(device), x_test)

                total_test_activation_loss += (torch.norm((torch.matmul(tensor_wrapper.get_tensor().to(device), test_random_x) - x_test), dim=0) ** 2).mean().item()

            tensor_factorizations_loss_histories["activation loss"][
                f"{factorization_label} trained using tensor loss"] = activation_loss_history
            tensor_factorizations_loss_histories["tensor loss"][
                f"{factorization_label} trained using tensor loss"] = tensor_loss_history
            if tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss":
                tensor_factorizations_loss_histories["penalization term"][
                    f"{factorization_label} trained using tensor loss"] = penalization_term_history

            # Save details to CSV
            csv_data.append({
                "Factorization": factorization_label,
                "Initial Activation Loss": initial_activation_loss,
                "Final Activation Loss": final_activation_loss,
                "Initial Tensor Loss": initial_tensor_loss,
                "Final Tensor Loss": final_tensor_loss,
                "Test Activation Loss": total_test_activation_loss.item(),
                "Initial Penalization Term": initial_penalization_term,
                "Final Penalization Term": penalization_term_history[-1] if tensor_factorizations_losses[tensor_factorization_index] == "penalized tensor loss" else None,
                "Training Time": time_elapsed,
                "Initialization Time": tensor_init_time
            })

        # Save the timing results to a file
        time_log_path = os.path.join(directory_path, file_name_no_format + "_training_times.txt")
        with open(time_log_path, "w") as f:
            f.writelines(time_log)

        # Save the loss results to a CSV file
        csv_path = os.path.join(directory_path, file_name_no_format + "_losses.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Factorization", "Initial Activation Loss", "Final Activation Loss", "Initial Tensor Loss",
                          "Final Tensor Loss", "Test Activation Loss", "Initial Penalization Term",
                          "Final Penalization Term", "Training Time", "Initialization Time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

        with open(file_path, "wb") as f:
            pkl.dump((tensor_factorizations_loss_histories, tensor_factorizations_dict), f)

    for label, activation_loss_history in tensor_factorizations_loss_histories["activation loss"].items():
        axes[0, 0].plot(activation_loss_history, label=f"{label}")
        axes[1, 0].plot(activation_loss_history[epoch_cut:], label=f"{label}")
        if configuration.contains("ylim"):
            axes[1, 0].set_ylim(0, configuration.get("ylim"))
        print(f"Activation loss for {label}: {activation_loss_history[-1]}")

    axes[0, 0].set_title("Full activation training loss history (target activation - approximated activation)")
    axes[1, 0].set_title(
        f"Loss history starting from epoch {epoch_cut} (target activation - approximated activation)")

    for label, tensor_loss_history in tensor_factorizations_loss_histories["tensor loss"].items():
        axes[0, 1].plot(tensor_loss_history, label=f"{label}", )
        axes[1, 1].plot(tensor_loss_history[epoch_cut:], label=f"{label}")
        if configuration.contains("ylim"):
            axes[1, 1].set_ylim(0, configuration.get("ylim"))
        print(f"Target tensor - approximated tensor for {label}: {tensor_loss_history[-1]}")

    axes[0, 1].set_title("Full loss history (target tensor - approximated tensor)")
    axes[1, 1].set_title(f"Loss history zoomed in from epoch {epoch_cut} (target tensor - approximated tensor)")

    for label, penalization_term_history in tensor_factorizations_loss_histories["penalization term"].items():
        axes[0, 2].plot(penalization_term_history, label=f"{label}")
        axes[1, 2].plot(penalization_term_history[epoch_cut:], label=f"{label}")
        print(f"Activation loss for {label}: {penalization_term_history[-1]}")

    axes[0, 2].set_title("Full penalization term history (sum of difference between matrices)")
    axes[1, 2].set_title(f"Penalization term history starting from epoch {epoch_cut} (sum of difference between matrices)")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    output_path = os.path.join(directory_path, file_name_no_format + "_plot.png")
    plt.savefig(output_path)
    plt.show()
