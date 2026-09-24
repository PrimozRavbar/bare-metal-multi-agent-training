import torch

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def format_for_qwen(example):
    parts = []

    for msg in example["messages"]:
        parts.append(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
        )

    return {
        "text": "\n".join(parts)
    }


def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
    )


class AgentDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            return_tensors="pt"
        )

        labels = torch.full_like(batch["input_ids"], -100)

        for i, ids in enumerate(batch["input_ids"]):
            full_ids = ids.tolist()
            text = self.tokenizer.decode(full_ids)
            pos = 0

            while True:
                start = text.find("<|im_start|>assistant\n", pos)

                if start == -1:
                    break

                start += len("<|im_start|>assistant\n")
                end = text.find("<|im_end|>", start)

                if end == -1:
                    break

                assistant_span = text[start:end]

                span_ids = self.tokenizer(
                    assistant_span,
                    add_special_tokens=False
                )["input_ids"]

                for j in range(len(full_ids) - len(span_ids) + 1):
                    if full_ids[j:j+len(span_ids)] == span_ids:
                        labels[i][j:j+len(span_ids)] = batch["input_ids"][i][j:j+len(span_ids)]
                        break

                pos = end

        batch["labels"] = labels
        return batch


def create_trainer(model, tokenizer, tokenized_dataset):

    data_collator = AgentDataCollator(tokenizer)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir="./qwen-agent-sft",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        #learning_rate=2e-4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        max_length=512,
        report_to="none",
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    return trainer
