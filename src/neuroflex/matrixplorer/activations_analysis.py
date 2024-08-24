import pickle as pkl

from neuroflex.utils.classification import load_original_model_for_sequence_classification, IMDBDataModule, \
    load_tokenizer_for_sequence_classification

from neuroflex.utils.printing_utils.printing_utils import Verbose
from neuroflex.utils.experiment_pipeline import Config

from neuroflex.utils.rank_analysis.utils import extract_based_on_path, AnalysisModelWrapper


def perform_activations_analysis(
        configuration: Config,
        verbose: Verbose
) -> None:
    """
    Performs the activations' analysis.

    Args:
        configuration (Config):
            The configuration object containing the necessary information to perform the analysis.
        verbose (Verbose):
            The verbosity level of the analysis.
    """

    # Getting the parameters related to the paths from the configuration
    file_available = configuration.get("file_available")
    file_path = configuration.get("file_path")
    directory_path = configuration.get("directory_path")
    file_name_no_format = configuration.get("file_name_no_format")

    # Getting the parameters related to the analysis from the configuration
    fig_size = configuration.get("figure_size") if configuration.contains("figure_size") else (20, 20)

    # Load the data if the file is available, otherwise process the model
    if file_available:
        print(f"The file '{file_path}' is available.")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
    else:
        # Loading and wrapping the model
        model = load_original_model_for_sequence_classification(configuration)
        tokenizer = load_tokenizer_for_sequence_classification(configuration)
        model_wrapper = AnalysisModelWrapper(model, verbose=verbose)

        # Loading the dataset
        dataset = IMDBDataModule(
            tokenizer=model_wrapper.tokenizer,
            max_len=model_wrapper.max_len,
            batch_size=model_wrapper.batch_size,
            num_workers=model_wrapper.num_workers,
            seed=model_wrapper.seed
        )
