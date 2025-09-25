# Purpose
# This script compares the output of the original (pretrained) TinyLlama model and your fine-tuned model on a set of test prompts, so you can see the effect of your local fine-tuning step-by-step.
#

#
# Import Libraries
# torch: For tensor operations and to run model inference.
# transformers: To load pre-trained and fine-tuned models and tokenizer.
#
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 
# Define Model and Paths
# Set the Hugging Face model ID and local directory where the fine-tuned model is saved.
#
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
finetuned_dir = "./finetuned-tinyllama"

#
# Load Tokenizer
# Loads the tokenizer using the same model ID as used for training. This ensures the input formatting matches what the model expects.
#
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 
# Helper Function: chat()
# Given a model, tokenizer, and a prompt, this function formats the prompt in the expected instruction style, encodes it, generates a response, and decodes the result back to text. It ensures only the generated response (not the prompt) is returned.
# 
def chat(model, tokenizer, prompt):
    input_text = f"### Instruction:\n{prompt}\n### Response:\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=32)
    output = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output.strip()

# 
# Test Prompts
# A short list of prompts is used to test both the original and the fine-tuned models, so their outputs can be easily compared.
# 
test_prompts = [
    "Who wrote Hamlet?",
    "Is Elon Musk still in Trump adminstration?",
    "What was the latest voting in Geneva in 2025?",
    "Where the US & EU tariffs negotiatios happens?",
]

# 
# Run the Original Model
# Loads the pretrained (unmodified) TinyLlama model and prints its responses to each test prompt.
# 
print("==== Responses from original (pretrained) model ====")
orig_model = AutoModelForCausalLM.from_pretrained(model_id)
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print("Response:", chat(orig_model, tokenizer, prompt))
    print("-" * 30)

#
# Run the Fine-Tuned Model
# Loads the locally fine-tuned model from disk and prints its responses to the same test prompts, so you can see if/where the model has “memorized” your custom answers.
#
print("\n==== Responses from finetuned model ====")
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_dir)
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print("Response:", chat(finetuned_model, tokenizer, prompt))
    print("-" * 30)

# Compare Outputs
# Visually compare how the answers change. If fine-tuning was effective, the new responses should match your fine-tuning data exactly (especially for your small training set), while the original answers may be more generic or less accurate.

# Tips
# Change or add test prompts to see the scope of fine-tuning.
# Try prompts outside your training data to see if the model generalizes.
# This workflow works for any PEFT/LoRA fine-tuned LLM in Hugging Face format.