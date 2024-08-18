from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model

from neuroflex import GlobalBaseModel
from neuroflex.utils import Config
from neuroflex.utils.storage_utils import store_model_and_info, load_model_and_info
from neuroflex.utils.chatbot.conversation_utils import load_original_model_for_causal_lm


config = Config("/Users/enricosimionato/Desktop/Alternative-Model-Architectures/src/experiments/configurations/CONFIG_BERT_CHAT.json")
config.start_experiment()

model = load_original_model_for_causal_lm(config)
tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_id"))


def test_store_model_and_info():
    store_model_and_info(
        model,
        config,
        tokenizer,
        verbose=True
    )


def test_store_model_and_info_peft():
    # Defining a LoRA configuration for training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(
        model,
        lora_config
    )

    store_model_and_info(
        peft_model,
        config,
        tokenizer,
        verbose=True
    )


def test_store_model_and_info_gdm():
    print(model)
    global_base_model = GlobalBaseModel(
        model,
        target_layers={
            "word_embeddings": {"rank": 128},
            "query": {"rank": 64},
            "key": {"rank": 64},
            "value": {"rank": 64},
            "dense": {"rank": 64},
        },
        use_names_as_keys=True,
        remove_average=True,
        preserve_original_model=False,
        verbose=True
    )

    store_model_and_info(
        global_base_model,
        config,
        tokenizer,
        verbose=True
    )


if __name__ == "__main__":
    #test_store_model_and_info()
    #test_store_model_and_info_peft()
    test_store_model_and_info_gdm()