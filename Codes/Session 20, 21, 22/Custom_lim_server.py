import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from openai import OpenAI
import uvicorn

# -------------------------
# 🔐 Hugging Face Login
# -------------------------
HF_TOKEN = "your_huggingface_token_here"
login(HF_TOKEN)

# -------------------------
# 🤖 Load Hugging Face Model
# -------------------------
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="./llm_models",
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./llm_models",
    token=HF_TOKEN
)

# -------------------------
# 🚀 FastAPI App
# -------------------------
app = FastAPI()

class RequestData(BaseModel):
    prompt: str
    max_length: int = 50


# -------------------------
# 🧠 Hugging Face Generation Endpoint
# -------------------------
@app.post("/generate")
async def generate_text(data: RequestData):
    inputs = tokenizer(data.prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=data.max_length
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": result}


# -------------------------
# 🌟 Gemini Endpoint
# -------------------------
@app.post("/geminiask")
async def generate_ans(data: RequestData):

    client = OpenAI(
        api_key="your_google_api_key_here",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data.prompt}
        ]
    )

    return {
        "generated_text": response.choices[0].message.content
    }


# -------------------------
# ▶ Run Server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)