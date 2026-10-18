# Chatbot evaluator

Human-in-the-loop chatbot answer evaluation with groundedness benchmarking,
evidence validation, local Ollama support, and a Google Colab QLoRA bootstrap
pipeline.

## Open the training notebook in Colab

[Open `colab_grounding_train.ipynb` in Google Colab](https://colab.research.google.com/github/Kapilpaliwal42/chatbot-evaluator/blob/main/training/colab_grounding_train.ipynb)

The notebook uses public Hugging Face grounding data to train a specialist
adapter. It does not replace the deterministic citation checks, completeness
checks, or human-review queue.

## Local quick start

```powershell
python evaluate.py --demo --out evaluation/output/demo
python review_app.py --cases evaluation/benchmark/v1/blind_review.jsonl
```

See [`evaluation/README.md`](evaluation/README.md) and
[`training/README.md`](training/README.md) for the full workflow.
