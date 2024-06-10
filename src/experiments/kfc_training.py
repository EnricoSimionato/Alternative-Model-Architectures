import torch

from transformers import AutoTokenizer

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

from gbm.utils.pipeline.config import Config
from gbm.utils.pipeline.experiment import Experiment
from gbm.utils.chatbot import OpenAssistantGuanacoDataModule
from gbm.utils.chatbot.conversation_utils import load_original_model_for_causal_lm


if __name__ == "__main__":
    configuration = Config(
        "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_KFC_CHAT.json"
    )

    original_model = load_original_model_for_causal_lm(configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        configuration.get("tokenizer_id_for_chat_template")).chat_template

    model = prepare_model_for_kbit_training(original_model)
    model = get_peft_model(model, LoraConfig(
        r=16,
        target_modules="all-linear"
    ))

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.requires_grad = True

    experiment = Experiment(
        task="chatbot",
        model=original_model,
        dataset=OpenAssistantGuanacoDataModule(
            configuration.get("batch_size"),
            configuration.get("num_workers"),
            tokenizer,
            configuration.get("max_len_tokenizer"),
            configuration.get("split")
        ),
        config=configuration,
        tokenizer=tokenizer
    )