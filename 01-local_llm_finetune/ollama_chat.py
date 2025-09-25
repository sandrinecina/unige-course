import requests

# Choose your model: "llama3" or "llama3-custom-lora" (if you created a LoRA)
MODEL_NAME = "llama3.2"

prompt = "Explain the benefits of renewable energy."
data = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": prompt}]
}

print(f"Sending to Ollama model: {MODEL_NAME}")
response = requests.post("http://localhost:11434/api/chat", json=data)
print("Response:")
print(response.json())
