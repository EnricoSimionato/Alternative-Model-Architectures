from transformers import AutoTokenizer

from gbm.utils.pipeline.experiment import Experiment
from gbm.utils.pipeline.config import Config
from gbm.utils.chatbot.conversation_utils import load_original_model_for_causal_lm
from gbm.utils.chatbot import OpenAssistantGuanacoDataModule


def test_experiment_with_chatbot():
    configuration = Config(
        "/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_LOCAL_BERT.json"
    )
    model = load_original_model_for_causal_lm(configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration.get("tokenizer_id"))
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        configuration.get("tokenizer_id_for_chat_template")).chat_template

    experiment = Experiment(
        task="chatbot",
        model=model,
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

    experiment.run_experiment()


if __name__ == "__main__":
    test_experiment_with_chatbot()