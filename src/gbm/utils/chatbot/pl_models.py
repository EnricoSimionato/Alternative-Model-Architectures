from __future__ import annotations

from typing import Any

import numpy as np

import torch

import pytorch_lightning as pl

import transformers

from gbm.utils.chatbot.conversation_utils import (
    get_conversation_example_1,
    get_conversation_example_2,
    start_conversation_loop
)
from gbm.utils.pl_utils.utility_mappings import optimizers_mapping


# TODO change the loss in a metric and update the description of the model
class CausalLMModelWrapper(pl.LightningModule):
    """
    Wrapper to train a CausalLMModel with Pytorch Lightning.

    Args:
        model (transformers.AutoModelForCausalLM):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        learning_rate (float):
            Learning rate. Defaults to 1e-5.
        max_epochs (int):
            Maximum number of epochs. Defaults to 3.
        warmup_steps (int):
            Number of warmup steps. Defaults to 0.
        kwargs:
            Additional keyword arguments.

    Attributes:
        model (transformers.AutoModelForCausalLM):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        learning_rate (float):
            Learning rate.
        max_epochs (int):
            Maximum number of epochs.
        warmup_steps (int):
            Number of warmup steps.
        training_step_index (int):
            Index of the training step.
        loss_history (dict[str, list[float]]):
            History of the losses.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
    """

    weights_to_exclude = [
        "lora",
        "vera"
    ]

    def __init__(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
        optimizers_settings: list[dict] = None,
        max_steps: int = 1,
        stop_tokens: list[str] = ("[INST]", "</s>"),
        kfc_training: bool = False,
        initial_regularization_weight: float = 0.01,
        max_regularization_weight: float = 10.0,
        start_step_regularization: int = 0,
        steps_regularization_weight_resets: int = 1000,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.optimizers_settings = optimizers_settings
        self.max_steps = max_steps

        self.stop_tokens = stop_tokens

        self.kfc_training = kfc_training
        self.initial_regularization_weight = initial_regularization_weight
        self.fixed_regularization_weight = None
        self.adaptive_regularization_weight = torch.tensor(
            initial_regularization_weight,
            requires_grad=False
        )
        self.max_regularization_weight = torch.tensor(
            max_regularization_weight,
            requires_grad=False
        )
        self.start_step_regularization = start_step_regularization
        self.steps_regularization_weight_resets = steps_regularization_weight_resets

        self.training_step_index = 0

        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0
        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0
        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

        self.loss_history = {
            "train": [],
            "validation": [],
            "test": []
        }

        self.model_dtype = dtype

    def configure_optimizers(
            self,
            **kwargs
    ) -> dict[str, torch.optim.Optimizer | str | Any]:
        """
        Configures the optimizer.

        Args:
            **kwargs:
                Additional keyword arguments.

        Returns:
            list[dict[str, torch.optim.Optimizer | str | Any]]:
                List of dictionaries containing the optimizer and the learning rate scheduler.
        """

        if self.optimizers_settings is None or self.optimizers_settings == []:
            self.optimizers_settings = [
                {
                    "optimizer": "adamw",
                    "parameters_group": [
                        name
                        for name, param in self.model.named_parameters()
                    ],
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "lr_scheduler": "cosine_with_warmup",
                    "warmup_steps": 0,
                    "monitored_metric": "loss"
                }
            ]
        if not all(key in optimizer_settings for key in ["optimizer", "parameters_group", "learning_rate"] for
                   optimizer_settings in self.optimizers_settings):
            raise ValueError(
                "The optimizers' settings are not well defined, they should contain the keys 'optimizer', 'parameters_group' and 'learning_rate'")
        if not all(optimizer_settings["optimizer"].lower() in optimizers_mapping for optimizer_settings in
                   self.optimizers_settings):
            raise ValueError(
                f"The following optimizers are not supported: {set(optimizer_settings['optimizer'] for optimizer_settings in self.optimizers_settings if optimizer_settings['optimizer'].lower() not in optimizers_mapping)}")

        optimizers = []
        for optimizer_settings in self.optimizers_settings:
            optimizer = optimizers_mapping[optimizer_settings["optimizer"].lower()](
                [param for name, param in self.model.named_parameters() if
                 name in optimizer_settings["parameters_group"]],
                lr=optimizer_settings["learning_rate"],
                eps=1e-7 if self.model_dtype == "float16" else 1e-8
            )

            if "lr_scheduler" in optimizer_settings:
                # TODO: Add the possibility to use different learning rate schedulers
                # TODO: Pass to optimizer and lr_scheduler a dictionary of parameters
                new_optimizer = {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": transformers.get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=optimizer_settings["warmup_steps"],
                            num_training_steps=self.max_steps,
                            num_cycles=0.5
                        ),
                        "monitor": optimizer_settings["monitored_metric"]
                    }
                }
            else:
                new_optimizer = {
                    "optimizer": optimizer
                }

            optimizers.append(new_optimizer)

        return optimizers

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor):
                Input IDs.
            kwargs
                Additional keyword arguments.

        Returns:
            torch.Tensor:
                Model outputs.
        """

        return self.model(input_ids, **kwargs)

    def get_unweighted_penalization(
            self
    ) -> torch.Tensor:
        """
        Computes the unweighted penalization term as the L1 norm of the model weights.

        Returns:
            torch.Tensor:
                Unweighted penalization term.
        """

        with torch.no_grad():
            original_params = []
            for name, param in self.model.named_parameters():
                if not any(substring in name for substring in CausalLMModelWrapper.weights_to_exclude):
                    original_params.append(param)

            sum_l1_norms = torch.tensor(0.0, device=self.device)
            for i, param in enumerate(original_params):
                sum_l1_norms += torch.sum(torch.abs(param.flatten()))

            return sum_l1_norms

    def get_weighted_penalization(
            self,
            penalization: torch.Tensor,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the weighted penalization term.

        Args:
            penalization (torch.Tensor):
                Unweighted penalization term.
            loss (torch.Tensor):
                Loss of the model computed for the current batch.

        Returns:
            torch.Tensor:
                Weighted penalization term.
        """

        if self.fixed_regularization_weight is None:
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
        elif (self.steps_regularization_weight_resets > 0 and
              self.training_step_index % self.steps_regularization_weight_resets == 0):
            self.fixed_regularization_weight = torch.tensor(
                2 * (loss / penalization).clone().detach().item(),
                requires_grad=False
            )
            self.adaptive_regularization_weight = torch.tensor(
                self.initial_regularization_weight,
                requires_grad=False
            )
            print("Fixed regularization weight reset to", self.fixed_regularization_weight.item(), "and adaptive regularization weight reset to", self.adaptive_regularization_weight.item())

        self.log(
            "fixed_regularization_weight",
            self.fixed_regularization_weight,
            on_step=True,
            on_epoch=False
        )

        return penalization * self.fixed_regularization_weight

    def regularization_scheduler_step(
            self
    ):
        """
        Updates the regularization weight.
        """

        k = torch.sqrt(torch.tensor(
            1.001,
            requires_grad=False
        )).to(self.adaptive_regularization_weight.device)
        self.adaptive_regularization_weight = self.adaptive_regularization_weight * k

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered train batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        if self.kfc_training and self.training_step_index >= self.start_step_regularization:
            self.log(
                "task_loss",
                loss,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "regularization_weight",
                self.regularization_weight,
                on_step=True,
                on_epoch=False,
            )

            unweighted_penalization = self.get_unweighted_penalization()
            self.log(
                "unweighted_penalization",
                unweighted_penalization,
                on_step=True,
                on_epoch=False
            )

            weighted_penalization = self.get_weighted_penalization(unweighted_penalization, loss)
            self.log(
                "weighted_penalization",
                unweighted_penalization,
                on_step=True,
                on_epoch=False
            )

            weighted_penalization = weighted_penalization.to(loss.device)
            self.adaptive_regularization_weight = self.adaptive_regularization_weight.to(loss.device)

            loss = loss + self.adaptive_regularization_weight * weighted_penalization

            self.regularization_scheduler_step()

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )

        self.log(
            "training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.training_step_losses_sum += loss.detach().cpu().numpy()
        self.training_step_losses_count += 1

        self.training_step_index += 1

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a validation step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered validation batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.validation_step_losses_sum += loss.detach().cpu().numpy()
        self.validation_step_losses_count += 1

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a test step.

        Args:
            batch (dict[str, torch.Tensor]):
                Batch of input data.
            batch_idx (int):
                Index of the batch.

        Returns:
            torch.Tensor:
                Loss of the model computed for the current batch.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Computing the loss of the model for the considered test batch
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True
        )

        self.test_step_losses_sum += loss.detach().cpu().numpy()
        self.test_step_losses_count += 1

        return loss

    def on_train_epoch_end(
            self
    ) -> None:
        """
        Performs operations at the end of a training epoch
        """

    def on_validation_epoch_end(
        self
    ) -> None:
        """
        Computes and stores the average training loss on the samples considered from the previous
        validation check to the current one and the average loss on the validation set.
        """

        if self.training_step_losses_count > 0:
            avg_train_loss = self.training_step_losses_sum / self.training_step_losses_count
            self.loss_history["train"].append(avg_train_loss)
        else:
            avg_train_loss = np.nan

        self.training_step_losses_sum = 0
        self.training_step_losses_count = 0

        if self.validation_step_losses_count > 0:
            avg_val_loss = self.validation_step_losses_sum / self.validation_step_losses_count
            self.loss_history["validation"].append(avg_val_loss)
        else:
            avg_val_loss = np.nan

        self.validation_step_losses_sum = 0
        self.validation_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_train_loss):
            print("Number of training steps equal to 0")
        print(f'Training loss: {avg_train_loss}')
        if np.isnan(avg_val_loss):
            print("Number of validation steps equal to 0")
        print(f'Validation loss: {avg_val_loss}')
        print("----------------------------------------------------------")

        self.start_conversation_trial()

    def on_test_epoch_end(
        self
    ) -> None:
        """
        Computes and stores the average validation on the test set.
        """

        if self.test_step_losses_count > 0:
            avg_test_loss = self.test_step_losses_sum / self.test_step_losses_count
            self.loss_history["test"].append(avg_test_loss)
        else:
            avg_test_loss = np.nan

        self.test_step_losses_sum = 0
        self.test_step_losses_count = 0

        print("----------------------------------------------------------")
        if np.isnan(avg_test_loss):
            print("Number of test steps equal to 0")
        print(f'Test loss: {avg_test_loss}')
        print("----------------------------------------------------------")

    def start_conversation_trial(
        self
    ) -> None:
        """
        Starts a conversation trial.
        """

        # Starting conversation loop
        dialogue_1 = start_conversation_loop(
            self.model,
            self.tokenizer,
            stop_tokens=self.stop_tokens,
            user_inputs=get_conversation_example_1(),
            make_model_trainable=True
        )

        # Starting conversation loop
        dialogue_2 = start_conversation_loop(
            self.model,
            self.tokenizer,
            stop_tokens=self.stop_tokens,
            user_inputs=get_conversation_example_2(),
            make_model_trainable=True
        )


class ChatbotModelWrapper(CausalLMModelWrapper):
    """
    Wrapper to train a model that is a chatbot with Pytorch Lightning.

    Args:
        model (transformers.AutoModelForCausalLM):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        learning_rate (float):
            Learning rate. Defaults to 1e-5.
        max_epochs (int):
            Maximum number of epochs. Defaults to 3.
        warmup_steps (int):
            Number of warmup steps. Defaults to 0.
        kwargs:
            Additional keyword arguments.

    Attributes:
        model (transformers.AutoModelForCausalLM):
            The model to wrap.
        tokenizer (transformers.AutoTokenizer | transformers.PreTrainedTokenizer):
            Tokenizer object.
        learning_rate (float):
            Learning rate.
        max_epochs (int):
            Maximum number of epochs.
        warmup_steps (int):
            Number of warmup steps.
        training_step_index (int):
            Index of the training step.
        loss_history (dict[str, list[float]]):
            History of the losses.
        training_step_losses_sum (float):
            Sum of the training step losses.
        training_step_losses_count (int):
            Number of training step losses.
        validation_step_losses_sum (float):
            Sum of the validation step losses.
        validation_step_losses_count (int):
            Number of validation step losses.
        test_step_losses_sum (float):
            Sum of the test step losses.
        test_step_losses_count (int):
            Number of test step losses.
    """

    def __init__(
            self,
            model: transformers.AutoModelForCausalLM,
            tokenizer: transformers.AutoTokenizer | transformers.PreTrainedTokenizer,
            learning_rate: float = 1e-5,
            max_steps: int = 1,
            warmup_steps: int = 0,
            stop_tokens: list[str] = ("[INST]", "</s>"),
            kfc_training: bool = False,
            initial_regularization_weight: float = 0.01,
            max_regularization_weight: float = 10000.0,
            dtype: torch.dtype = torch.float32,
            **kwargs
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            learning_rate,
            max_steps,
            warmup_steps,
            stop_tokens,
            kfc_training,
            initial_regularization_weight,
            max_regularization_weight,
            dtype,
            **kwargs
        )



