import pickle as pkl

from neuroflex.utils.classification import load_original_model_for_sequence_classification, IMDBDataModule, \
    load_tokenizer_for_sequence_classification

from neuroflex.utils.experiment_pipeline import Config

from neuroflex.utils.rank_analysis.utils import extract_based_on_path, AnalysisModelWrapper


def perform_activations_analysis(
        configuration: Config,
) -> None:
    """
    Performs the activations' analysis.

    Args:
        configuration (Config):
            The configuration object containing the necessary information to perform the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    # Getting the parameters related to the analysis from the configuration
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)

    # Getting the parameters related to the model and the data from the configuration
    batch_size = configuration.get("batch_size") if configuration.contains("batch_size") else 64
    num_workers = configuration.get("num_workers") if configuration.contains("num_workers") else 1
    seed = configuration.get("seed") if configuration.contains("seed") else 42
    max_len = configuration.get("max_len") if configuration.contains("max_len") else 512

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        # Loading and wrapping the model
        model = load_original_model_for_sequence_classification(configuration)
        tokenizer = load_tokenizer_for_sequence_classification(configuration)
        model_wrapper = AnalysisModelWrapper(model)

        # Loading the dataset
        dataset = IMDBDataModule(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            split=(0.8, 0.1, 0.1),
            seed=seed
        )

        # Performing the activation analysis
        data_loader = dataset.train_dataloader()
        for batch in data_loader:
            y = model_wrapper.forward(batch)