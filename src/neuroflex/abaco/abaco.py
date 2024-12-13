from enum import Enum

import torch

import transformers

from exporch import Config
from exporch.wrapping.model_wrapper import ModelWrapper

from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model


class AlphaStrategy(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ABACOModel(ModelWrapper):
    """
    Model wrapper that allows to perform Adapter-Based Approximation and Compression Optimization method to compress a
    pre-trained model.
    ABACO aims at moving the information from pre-trained weights to low-rank adapters put on top of the model.
    The method performs fine-tuning in such a way that pre-trained parameters become less relevant for the output
    computation, and the adapters take over the main role.

    Attributes:
        model (peft.PeftModel | peft.PeftMixedMode):
            Model with adapters on top of it.
        initial_alpha (float):
            Initial value of the alpha parameter.
        alpha (float):
            Alpha parameter.
            Alpha parameter is used to balance the contribution of the pre-trained model and the adapters.
            During fine-tuning, the alpha parameter is updated to make the pre-trained model less relevant.
        horizon (int):
            Number of steps of fine-tuning that will be performed.
            Depending on the alpha strategy, the alpha parameter will approach 0 differently.
        alpha_strategy (AlphaStrategy):
            Strategy to update the alpha parameter.

    Args:
        model (transformers.AutoModel | transformers.PreTrainedModel)
            Model with adapters on top of it.
        config (Config):
            Configuration object that contains additional information.
        initial_alpha (float):
            Initial value of the alpha parameter.
        horizon (int):
            Number of steps of fine-tuning that will be performed.
            Depending on the alpha strategy, the alpha parameter will approach 0 differently.
        alpha_strategy (AlphaStrategy):
            Strategy to update the alpha parameter.
    """

    def __init__(
            self,
            model: transformers.AutoModel | transformers.PreTrainedModel,
            config: Config,
            initial_alpha: float = 1.0,
            horizon: int = 10000,
            alpha_strategy: str = "exponential"
    ) -> None:
        super().__init__()

        # Wrapping the model with the adapters
        self.model = get_adapted_model(model, config)

        # TODO For now the code is explicitly using LoRA
        # Making the adapters trainable
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.initial_alpha = initial_alpha
        self.alpha = initial_alpha
        self.horizon = horizon
        self.alpha_strategy = AlphaStrategy(alpha_strategy)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            **kwargs
    ) -> None:
        """
        Forward pass of the model.

        Args:
            input_ids:
                Input IDs.
            attention_mask:
                Attention mask.

        Returns:
            Output tensor.
        """

        output = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return output

    def before_training_step(
            self,
            training_step: int
    ) -> None:
        """
        Method to call before the training step to take some operations.
        In the case of ABACO, it updates the alpha parameter of the underlying model.

        Args:
            training_step (int):
                Current training step.
        """

        # Setting the current alpha parameter to the model
        self.model.set_alpha(self.alpha)

        # Updating the alpha parameter
        self.update_alpha(training_step)

    def update_alpha(
            self,
            training_step: int
    ) -> None:
        """
        Updates the alpha parameter.

        Args:
            training_step (int):
                Current training step.
        """

        if self.alpha_strategy == AlphaStrategy.LINEAR:
            self.alpha = max(0.0, self.initial_alpha - training_step / self.horizon)
        elif self.alpha_strategy == AlphaStrategy.EXPONENTIAL:
            self.alpha = max(0.0, self.initial_alpha * 2 ** (- 5 * training_step / self.horizon))
        else:
            raise ValueError("Invalid update strategy for alpha.")

    @torch.no_grad()
    def get_logging_info(
            self
    ) -> list:
        """
        Returns additional information to log.

        Returns:
            list:
                Additional information to log.
        """

        return [
            {"name": "alpha", "value": self.alpha, "on_step": True, "on_epoch": False, "prog_bar": True},
        ]

    def factorization(
            self,
            value: bool
    ) -> None:
        """
        Factorizes the model.

        Args:
            value (bool):
                If True, the model is factorized: weights that are wrapped by the adapters are replaced by the wrappers
                themselves. If False, the model uses the current alpha to weight the contributions of the pre-trained
                weights.
        """

        if value:
            self.model.set_alpha(0.0)
        else:
            self.model.set_alpha(self.alpha)
