# Local Fine-Tuning Exercise: TinyLlama + PEFT

This exercise shows how to:

- Download and run a small LLM (TinyLlama) **locally**
- Fine-tune it on a few text pairs using **LoRA/PEFT**
- Compare model responses before and after fine-tuning

## 1. Install Requirements

```
pip install torch transformers peft datasets
```

## 2. Run Fine-Tuning

```
python train.py
```

- This fine-tunes TinyLlama on a small set of Q&A pairs.

## 3. Compare Model Responses

```
python compare.py
```

- See the difference in responses _before_ and _after_ fine-tuning.

## Notes

- This will run on CPU, and takes a few minutes on a laptop for the tiny dataset.
- You can edit the prompts/responses in `train.py` to try your own "personalization"!

## Model Card

- Uses [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — 1.1B param open model.

## My notes:

- When we fine-tune with 20 epochs (run our dataset 20x) we have a good results vs with 3 epochs we will have wrong answers, so it's not enought.
- In real case scenario, you can do fine-tuning directly in Amazon Bedrock.
