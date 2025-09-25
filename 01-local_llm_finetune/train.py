#transformers: Handles models and tokenization.
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# datasets: Loads/handles training data.
from datasets import Dataset 

#peft: 'Parameter-Efficient Fine-Tuning' (e.g., LoRA)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

#
#Loads the tokenizer and model weights from Hugging Face.
#- Tokenizer: Converts text ↔ tokens the model understands.
#- Model: The language model to be fine-tuned.
#
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

#
# Set Up LoRA Configuration
# LoRA: Instead of updating the full model, only tiny “adapters” are trained—fast, memory-efficient, and *safe* for laptops.
# target_modules: Should match your model architecture for best effect.
#
# Parameters:
# task_type = TaskType.CAUSAL_LM: Specifies we are fine-tuning a causal language model (like GPT).
# r = controls how many new parameters LoRA introduces. It is like attaching a small “learning pad” to the giant model, and r controls the size of that pad.
    ## Higher r -> more parameters -> the model can learn more patterns (but training takes more memory and compute).
    ## Lower r -> fewer parameters -> cheaper to train, but the model may not adapt as well.
# Lora_alpha = scales the updates from the LoRA layers. Higher values mean larger updates, which can help the model learn faster, but may also lead to instability if set too high.
    ## Controls how strongly the new parameters affect the original machine.
    ## Higher value -> parameters have more power, stronger effect.
    ## Lower value -> parameters are subtle, less influence.
    ## Think of it like turning up or down the volume of the LoRA tweaks.
# Lora_dropout = randomly “turns off” some LoRA parameters during training, which helps prevent overfitting (memorizing the training data too exactly).
    ## During training, sometimes we randomly ignore (drop) 5% of the parameters' input.
    ## This keeps the model from over-relying on a few parameters -> makes it more robust.
    ## If you set it too high, the model struggles to learn; too low, and it may overfit.
# target_modules = which parts of the model to apply LoRA to. Should match the architecture of your base model.
    ## For TinyLlama, we target the query and value projection layers in the attention mechanism.
    ## The model has many parts, but we don’t need parameters everywhere.
    ## We add them only to q_proj and v_proj, which are important for how the model pays attention to words.
    ## Fewer parts -> faster and lighter training.
    ## If we added parameters to more parts, we’d spend more compute but maybe learn a bit better.
    ## Descripes layers (paper to read: Attention Is All You Need) https://arxiv.org/abs/1706.03762
    
    ## How to get model layers:
        ### from transformers import AutoModelForCausalLM
        ### model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        ### print(model)    
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

# 
# Prepare Your Training Data
# - train_data: List of dictionaries, each with a prompt and target response.
# - Dataset: A Hugging Face utility for handling mini datasets.
# 
train_data = [
    {"prompt": "Is Elon Musk still in Trump adminstration?", "response": "No, he left the goverment on the 1st of June 2025."},
    {"prompt": "What was the latest voting in Geneva in 2025?", "response": "1) Solar energy adoption, 2) Cantonal Tax allocation , 3) Subletting regulations"},
    {"prompt": "Where the US & EU tariffs negotiatios happens?", "response": "In Paris, France."},
]
ds = Dataset.from_list(train_data)

#
# Preprocess: Convert Text to Tokens
# - Prompt Formatting: Matches the chat model’s style ("### Instruction...").
# - Tokenization: Converts text to integer IDs for the model.
# - labels: Set to input_ids (teacher forcing)—so the model is trained to output the full text sequence, including prompt/response.
#
def preprocess(example):
    text = f"### Instruction:\n{example['prompt']}\n### Response:\n{example['response']}"
    encoding = tokenizer(
        #text, truncation=True, padding="max_length", max_length=64, return_tensors="pt"
        text, truncation=True, padding="max_length", max_length=64
    )
    encoding["labels"] = encoding["input_ids"]
    return encoding

ds = ds.map(preprocess, remove_columns=ds.column_names)
ds.set_format(type="torch")

# 
# - Batch size: 1 (tiny dataset, no GPU required)
# - Epochs: How many times to cycle through your dataset (more = memorize better).
# - Learning rate: How much to adjust per step.
# - Output dir: Where to save logs/checkpoints.
# Parameters:
    ## per_device_train_batch_size=1: Since our dataset is tiny, we use a batch size of 1. This means the model updates its parameters after seeing each training example.
        ### This is how many training samples each GPU (or CPU) processes at once.
        ### A batch size of 1 = the model sees one example, updates its weights, then moves on.
        ### Small batch sizes use less memory but training can be noisy (weights jump around more).
        ### Larger batch sizes are smoother and faster on big GPUs, but need more memory.
    ## num_train_epochs=20: We train for 20 epochs, meaning the model sees the entire dataset 20 times. This helps it memorize our tiny dataset.
        ### One epoch = the model has gone through the entire training dataset once.
        ### 20 epochs means the model will pass over your dataset 20 times.
        ### More epochs = more learning opportunities, but risk of overfitting (model memorizes instead of generalizing).
    ## learning_rate=4e-4: The learning rate controls how big of a step the model takes during each update. 0.0004 is a moderate value for fine-tuning.
        ### If too high, the model may overshoot the best parameters and fail to learn.
        ### If too low, training will be very slow and may get stuck in suboptimal solutions
    ## output_dir="./output": Where to save model checkpoints and logs during training.
        ### Make sure this directory exists or the Trainer will create it for you.
    ## logging_steps=1: Log training progress every step (since dataset is tiny). We can come back to any step if needed.
        ### With a tiny dataset, we log every step to closely monitor training.
        ### For larger datasets, you might log every 100 or 500 steps instead.
    ## report_to="none": Disable reporting to external services (like WandB) for simplicity. 
        ### This keeps things simple for local training without external logging.
        ### For serious training, you might want to log to a service to track metrics over time.
#
args = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=18,
    learning_rate=4e-4,
    output_dir="./output",
    logging_steps=1,
    report_to="none",
)

    # Note: With epoch 20, we had overfitting: the model could not longer answer the question that was not in the training set! It just repeated the training answers.

# 
# Trainer: Train the Model!
# - Trainer: Hugging Face's simple training loop—handles all the boilerplate for you.
# - train(): Actually runs the fine-tuning.
# 
trainer = Trainer(model=model, train_dataset=ds, args=args)
trainer.train()

#
# 8. Save Your Fine-Tuned Model
# - Saves the adapter and tokenizer to disk so you can reload/use later.
#
model.save_pretrained("./finetuned-tinyllama")
tokenizer.save_pretrained("./finetuned-tinyllama")
print("Fine-tuning complete! Model saved in ./finetuned-tinyllama")

#
# After Fine-Tuning
# You can use your new model just like the base model:
# - Load with from_pretrained('./finetuned-tinyllama')
# - Generate text as before
# - Should now give you your domain-specific (or silly overfit) answers for the prompts you trained on


# Key Concepts
# - LoRA/PEFT: Trains only small “side modules” (adapters), so it’s cheap and efficient.
# - Prompt/Response formatting: Must match model style!
# - Overfitting is OK for demo: You want to see obvious changes.
#