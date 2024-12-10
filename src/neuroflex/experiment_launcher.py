import sys

from exporch import GeneralPurposeExperimentFactory

from neuroflex.experiments.abaco_experiment import ABACOExperiment
from neuroflex.experiments.factorization_experiment import FactorizationBenchmarkEvaluation, FactorizationFineTuningExperiment
from neuroflex.experiments.layer_replacement_experiment import (
    LayerReplacementFineTuningExperiment,
    LayerReplacementFineTuningEntireModelExperiment,
    LayerReplacementFineTuningAdapterOnTargetsExperiment,
    LayerReplacementFineTuningAdapterOnEntireModelExperiment
)

GeneralPurposeExperimentFactory.register({
    "factorization_benchmark_evaluation": FactorizationBenchmarkEvaluation,
    "factorization_fine_tuning_experiment": FactorizationFineTuningExperiment,

    "layer_replacement_fine_tuning_experiment": LayerReplacementFineTuningExperiment,
    "layer_replacement_fine_tuning_entire_model_experiment": LayerReplacementFineTuningEntireModelExperiment,
    "layer_replacement_fine_tuning_adapter_on_targets_experiment": LayerReplacementFineTuningAdapterOnTargetsExperiment,
    "layer_replacement_fine_tuning_adapter_on_entire_model_experiment": LayerReplacementFineTuningAdapterOnEntireModelExperiment,

    "abaco_experiment": ABACOExperiment

})


def main() -> None:
    """
    Main method to start the various types of experiments on a deep model.
    """

    if len(sys.argv) < 2:
        raise Exception("Please provide the name of the configuration file.\n"
                        "Example: python src/redhunter/analysis_launcher.py config_name")

    # Extracting the configuration name and the environment
    config_file_name = sys.argv[1]

    # Creating and launching the experiment
    experiment = GeneralPurposeExperimentFactory.create(f"src/experiments/configurations/{config_file_name}")
    experiment.launch_experiment()


if __name__ == "__main__":
    main()
