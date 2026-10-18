# Hugging Face bootstrap and Colab training

This pipeline uses public data to improve **groundedness detection**. It does
not pretend that public hallucination labels teach this project’s complete
`pass`/`fail`/`review` rubric. Instruction following, completeness, source
conflicts, and citation validity still require project-specific reviewed cases.

## Prepare public data

In Colab or a Python environment:

```powershell
pip install -U datasets
python training/prepare_hf_data.py `
  --dataset wandb/RAGTruth-processed `
  --out training/ragtruth-grounding.jsonl `
  --limit 10000
```

The normalizer skips rows whose schema cannot provide a question, context,
answer, and explicit grounding label. Inspect the generated manifest before
training. RAGTruth++ can be prepared the same way:

```powershell
python training/prepare_hf_data.py `
  --dataset blue-guardrails/ragtruth-plus-plus `
  --out training/ragtruth-plus-plus-grounding.jsonl
```

## Train in Google Colab

```python
!pip install -U transformers datasets peft trl bitsandbytes accelerate
!pip uninstall -y torchao
!git clone <your-repository-url> /content/project
%cd /content/project
!python training/prepare_hf_data.py --dataset wandb/RAGTruth-processed \
  --out training/ragtruth-grounding.jsonl --limit 10000
!python training/colab_train.py --data training/ragtruth-grounding.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct --out training/grounding-lora
```

The output is a LoRA adapter. Evaluate it on a held-out public split and on a
new project-specific set before connecting it to the evaluator. Do not use the
project test split or the frozen benchmark holdout for training or tuning.
