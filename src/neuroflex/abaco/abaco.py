from enum import Enum

import torch
from torch import device

import transformers

from exporch import Config
from exporch.utils import LoggingInterface
from neuroflex.factorization.factorized_model import FactorizedModel

from neuroflex.utils.adapters_utils.adapters_utils import get_adapted_model


class AlphaStrategy(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ABACOModel(torch.nn.Module, LoggingInterface, FactorizedModel):
    """
    Model wrapper that allows to perform KFC-alpha training in which the pre-trained weights become less relevant for
    the output computation as the training goes on.

    Args:
        model (transformers.PreTrainedModel)
            Model to wrap.
        initial_alpha (float):
            Initial value of the alpha parameter.
        horizon (int):
            Number of steps before the alpha parameter reaches the maximum.
        strategy (AlphaStrategy):
            Strategy to update the alpha parameter.
        **kwargs:
            Additional keyword arguments.

    Attributes:
        model (peft.PeftModel):
            Model with adapters to wrap.
        alpha (float):
            Alpha parameter.
        horizon (int):
            Number of steps before the alpha parameter reaches the maximum.
        alpha_strategy (AlphaStrategy):
            Strategy to update the alpha parameter.
    """

    def __init__(
            self,
            model: transformers.PreTrainedModel,
            config: Config,
            initial_alpha: float = 1.0,
            horizon: int = 10000,
            alpha_strategy: str = "exponential",
            **kwargs
    ) -> None:
        torch.nn.Module.__init__(self, **kwargs)
        LoggingInterface.__init__(self, **kwargs)
        FactorizedModel.__init__(self, **kwargs)

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

        self.model.set_alpha(
            self.alpha,
            **kwargs
        )

        self.update_alpha(
            training_step,
            **kwargs
        )

    def update_alpha(
            self,
            training_step: int,
            **kwargs
    ) -> None:
        """
        Updates the alpha parameter.

        Args:
            training_step (int):
                Current training step.
            **kwargs:
                Additional keyword arguments.
        """

        if self.alpha_strategy == AlphaStrategy.LINEAR:
            self.alpha = max(0.0, self.initial_alpha - training_step / self.horizon)
        elif self.alpha_strategy == AlphaStrategy.EXPONENTIAL:
            self.alpha = max(0.0, self.initial_alpha * 2 ** (- 5 * training_step / self.horizon))
        else:
            raise ValueError("Invalid alpha strategy.")

    @torch.no_grad()
    def get_logging_info(
            self,
            **kwargs
    ) -> list:
        """
        Returns additional information to log.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
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

