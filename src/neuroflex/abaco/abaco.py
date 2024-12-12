from enum import Enum

import torch
from torch import device

import transformers

from exporch import Config
from exporch.utils import LoggingInterface

from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model
from exporch.wrapping.model_wrapper import ModelWrapper


class AlphaStrategy(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ABACOModel(ModelWrapper):
    """
    Model wrapper that allows to perform KFC-alpha training in which the pre-trained weights become less relevant for
    the output computation as the training goes on.


    Attributes:
        model (peft.PeftModel | peft.PeftMixedMode):
            Model with adapters on top of it.
        initial_alpha (float):
            Initial value of the alpha parameter.
        alpha (float):
            Alpha parameter.
        horizon (int):
            Number of steps before the alpha parameter reaches the maximum.
        alpha_strategy (AlphaStrategy):
            Strategy to update the alpha parameter.

    Args:
        model (transformers.AutoModel | transformers.PreTrainedModel)
            Model to wrap.
        initial_alpha (float):
            Initial value of the alpha parameter.
        horizon (int):
            Number of steps before the alpha parameter reaches the maximum.
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

        # TODO To change the code to improve these lines
        config.set("target_modules", list(config.get("targets")))
        self.model = get_adapted_model(model, config)
        # just for now making the adapters trainable
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.initial_alpha = initial_alpha
        self.alpha = initial_alpha
        self.horizon = horizon
        self.alpha_strategy = AlphaStrategy(alpha_strategy)

    def get_model(
            self
    ) -> None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
        """
        Returns the model.

        Returns:
            None | torch.nn.Module | transformers.AutoModel | transformers.PreTrainedModel:
                Model.
        """

        return self.model

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
            training_step: int,
            **kwargs
    ) -> None:
        """
        Method to call before the training step to take some operations.
        In the case of KFC-alpha training, it updates the alpha parameter of the underlying model.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        self.model.set_alpha(self.alpha)
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
            raise ValueError("Invalid alpha strategy.")

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

    @property
    def device(self) -> device:
        """
        Device where the model is located.

        Returns:
            Device.
        """

        return next(self.parameters()).device
