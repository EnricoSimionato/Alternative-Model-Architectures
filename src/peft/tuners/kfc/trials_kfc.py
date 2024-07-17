import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import get_peft_model
from peft.tuners.kfc import KFCLoraLayer

from peft.tuners.kfc.config import KFCLoraConfig


def get_peft_model_kfc():
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    kfc_lora_config = KFCLoraConfig(
        r=128,
        target_modules="all-linear",
        bias="none"
    )

    kfc_lora_model = get_peft_model(pretrained_model, kfc_lora_config)

    print(kfc_lora_model)
    print(type(kfc_lora_model.base_model.model.bert.encoder.layer[4].attention.self.query))
    print(kfc_lora_model.base_model.model.bert.encoder.layer[4].attention.self.query.kfc_alpha)


def kfc_model():
    pass


def kfc_forward():
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    kfc_lora_config = KFCLoraConfig(
        r=128,
        target_modules="all-linear",
        bias="none"
    )

    kfc_lora_model = get_peft_model(pretrained_model, kfc_lora_config)
    for module in kfc_lora_model.model.modules():
        if isinstance(module, KFCLoraLayer):
            assert module.kfc_alpha == 1.0

    kfc_lora_model.set_alpha(0.5)
    for module in kfc_lora_model.model.modules():
        if isinstance(module, KFCLoraLayer):
            assert module.kfc_alpha == 0.5

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "This is a test input."
    inputs = tokenizer(text, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    kfc_lora_model.set_alpha(1.0)
    output = kfc_lora_model(input_ids=input_ids, attention_mask=attention_mask)
    print(output)

    kfc_lora_model.set_alpha(0.0)
    output = kfc_lora_model(input_ids=input_ids, attention_mask=attention_mask)
    print(output)

    print()


if __name__ == "__main__":
    kfc_forward()
