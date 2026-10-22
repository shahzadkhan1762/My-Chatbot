
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import torch

# Reduce CPU thread overhead on older processors.
CPU_THREADS = 2
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

app = FastAPI(
    title="My AI Agent API",
    description="API for my fine-tuned TinyLlama chatbot",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
)

tokenizer = None
model = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The AI model is not ready yet.",
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message.",
        )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer clearly and concisely.",
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][
            inputs["input_ids"].shape[1]:
        ]

        reply = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if not reply:
            reply = "I couldn't generate a response. Please try asking another way."

        return ChatResponse(reply=reply)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {exc}",
        ) from exc


@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print("Loading fine-tuned adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    print("AI Agent model loaded successfully!", flush=True)
