"""QLoRA training entrypoint for the normalized grounding dataset.

Designed for Google Colab. It trains a grounding specialist; the existing
evaluator remains responsible for instruction-following, completeness, citation
validation, and final human-review routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_example(row: dict) -> str:
    instruction = (
        "Classify whether the answer is supported by the supplied context. "
        "Return only JSON with a groundedness key whose value is supported or unsupported.\n\n"
        f"Question:\n{row['question']}\n\nContext:\n{row['context']}\n\nAnswer:\n{row['answer']}"
    )
    return f"<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{row['completion']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA-train a grounding specialist in Colab.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("grounding-lora"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install Colab dependencies first: pip install -U transformers datasets peft trl bitsandbytes accelerate"
        ) from exc

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 30:
        raise SystemExit("Refusing to train on fewer than 30 normalized examples.")
    dataset = Dataset.from_list([{"text": format_example(row)} for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=16,
        learning_rate=2e-4, logging_steps=10, save_strategy="epoch",
        bf16=True, gradient_checkpointing=True, report_to="none",
    )
    trainer = SFTTrainer(
        model=args.model, train_dataset=dataset, peft_config=lora,
        processing_class=tokenizer, args=training, dataset_text_field="text",
        max_seq_length=args.max_length,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Saved adapter to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
