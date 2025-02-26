import sys
from torch import device

from exporch import GeneralPurposeExperimentFactory

from neuroflex.experiments.abaco_experiment import ABACOExperiment
from neuroflex.experiments.factorization_experiment import FactorizationBenchmarkEvaluation, FactorizationFineTuningExperiment
from neuroflex.experiments.layer_replacement_experiment import (
    LayerReplacementFineTuningExperiment,
    LayerReplacementFineTuningEntireModelExperiment,
    LayerReplacementFineTuningAdapterOnTargetsExperiment,
    LayerReplacementFineTuningDifferentAdapterOnTargetsExperiment
)

GeneralPurposeExperimentFactory.register({
    "factorization_benchmark_evaluation": FactorizationBenchmarkEvaluation,
    "factorization_fine_tuning_experiment": FactorizationFineTuningExperiment,

    "layer_replacement_fine_tuning_experiment": LayerReplacementFineTuningExperiment,
    "layer_replacement_fine_tuning_entire_model_experiment": LayerReplacementFineTuningEntireModelExperiment,
    "layer_replacement_fine_tuning_adapter_on_targets_experiment": LayerReplacementFineTuningAdapterOnTargetsExperiment,
    "layer_replacement_fine_tuning_different_adapter_on_targets_experiment": LayerReplacementFineTuningDifferentAdapterOnTargetsExperiment,

    "abaco_experiment": ABACOExperiment
})
"""
def main() -> None:
    #""
    #Main method to start the various types of experiments on a deep model.
    #""
    if len(sys.argv) < 2:
        raise Exception("Please provide the name of the configuration file.\n"
                        "Example: python src/redhunter/analysis_launcher.py config_name")

    # Extracting the configuration name and the environment
    config_file_name = sys.argv[1]

    # Creating and launching the experiment
    experiment = GeneralPurposeExperimentFactory.create(f"src/experiments/configurations/{config_file_name}")
    experiment.launch_experiment()
"""
import torch
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch import Config

def main() -> None:
    if len(sys.argv) < 2:
        raise Exception("Please provide the name of the configuration file.\n"
                        "Example: python src/redhunter/analysis_launcher.py config_name")

    # Extracting the configuration name and the environment
    config_file_name = sys.argv[1]

    #num_layers = 12
    #path = "src/experiments/results/bert-base-uncased/factorization_fine_tuning_experiment/version_0/GlobalBase.pt"

    num_layers = 32
    #path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_92/LocalSVD.pt"
    path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_10/GlobalBase.pt"
    #path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_14/GlobalBase.pt"

    #path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_91/LocalSVD.pt"
    #path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_11/GlobalBase.pt"
    #path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_15/GlobalBase.pt"

    config = Config(path.replace("GlobalBase.pt", "config.yaml").replace("LocalSVD.pt", "config.yaml"))

    original_model = load_model_for_causal_lm(config)
    original_model.to("cpu")
    original_weights = []
    for i in range(num_layers):
        print(f"Layer {i}")
        #original_weights.append(original_model.bert.encoder.layer[i].attention.self.key.weight)
        #original_weights.append(original_model.model.layers[i].mlp.gate_proj.weight)
        original_weights.append(original_model.model.layers[i].mlp.up_proj.weight)

    with open(path, "rb") as f:
        model = torch.load(f, weights_only=False)
    model.to("cuda")
    weights = []
    for i in range(num_layers):
        print(f"Layer {i}")
        #weights.append(model.bert.encoder.layer[i].attention.self.key.weight)
        #weights.append(model.model.layers[i].mlp.gate_proj.weight.to("cpu"))
        weights.append(model.model.layers[i].mlp.up_proj.weight)
    del model
    del original_model
    torch.cuda.empty_cache()

    avg_sse = 0
    avg_rsse = 0

    for i in range(num_layers):
        print(f"Layer {i}")
        #print(original_weights[i])
        #print(weights[i])
        with torch.no_grad():
            original_weight = original_weights[i]
            approximated_weight = weights[i]
            original_weight = original_weight.to(torch.float32).to("cuda")
            approximated_weight = approximated_weight.to(torch.float32).to("cuda")

            sse = torch.sum((original_weight - approximated_weight) ** 2).to("cpu")
            rsse = torch.sum((original_weight - approximated_weight) ** 2) / torch.sum(original_weight ** 2).to("cpu")

            avg_sse += sse
            avg_rsse += rsse

    avg_sse /= num_layers
    avg_rsse /= num_layers

    print(f"path:{path}")

    print(f"Average SSE: {avg_sse}")
    print(f"Average RSSE: {avg_rsse}")

"""
import torch
from exporch.utils.causal_language_modeling import load_model_for_causal_lm
from exporch import Config

def main() -> None:
    if len(sys.argv) < 2:
        raise Exception("Please provide the name of the configuration file.\n"
                        "Example: python src/redhunter/analysis_launcher.py config_name")

    # Extracting the configuration name and the environment
    config_file_name = sys.argv[1]

    num_layers = 32
    #path_1 = "src/experiments/results/bert-base-uncased/factorization_fine_tuning_experiment/version_0/GlobalBase.pt"
    #path_2 = "src/experiments/results/bert-base-uncased/factorization_fine_tuning_experiment/version_1/GlobalBase.pt"
    #dest_path = "src/experiments/results/bert-base-uncased/factorization_fine_tuning_experiment/version_2/GlobalBase.pt"

    path_1 = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_14/GlobalBase.pt"
    path_2 = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_15/GlobalBase.pt"
    dest_path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_16/GlobalBase.pt"

    #path_1 = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_14/fine_tuned_model_GlobalBase.pt"
    #path_2 = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_15/GlobalBase.pt"
    #dest_path = "src/experiments/results/Llama-3.1-8B/factorization_fine_tuning_experiment/version_17/GlobalBase.pt"

    with open(path_1, "rb") as f:
        model_1 = torch.load(f, weights_only=False)
        print(model_1)
        print("#################################################################################################\n\n\n")

    with open(path_2, "rb") as f:
        model_2 = torch.load(f, weights_only=False)
        print(model_2)
        print("#################################################################################################\n\n\n")

    for i in range(num_layers):
        model_1.model.layers[i].mlp.up_proj = model_2.model.layers[i].mlp.up_proj
        #model_1.bert.encoder.layer[i].attention.self.key = model_2.bert.encoder.layer[i].attention.self.key

    print(model_1)
    with open(dest_path, "wb") as f:
        torch.save(model_1, f)
"""
if __name__ == "__main__":
    main()
