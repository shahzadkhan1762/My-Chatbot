
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    print("Loading your fine-tuned adapter...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print("\nYour AI Chatbot is ready!")
    print("Type 'exit' to close the chatbot.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Chatbot closed. Goodbye!")
            break

        if not question:
            continue

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        print("Bot is thinking...", flush=True)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        print(f"\nBot: {answer.strip()}\n")

if __name__ == "__main__":
    main()
