from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Muat model GPT-2 dan tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Pydantic model untuk input JSON
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 100  # Panjang teks maksimum yang ingin dihasilkan

# Fungsi untuk menghasilkan teks menggunakan GPT-2
def generate_text(prompt: str, max_length: int):
    # Tokenisasi prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Menghasilkan teks menggunakan model
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, top_k=60, temperature=0.7)

    # Decode output dan konversi ke string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Endpoint API untuk menghasilkan teks
@app.post("/generate-text/")
async def generate_text_endpoint(request: PromptRequest):
    prompt = request.prompt
    max_length = request.max_length
    generated_text = generate_text(prompt, max_length)
    return {"generated_text": generated_text}

#test commit
# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Welcome to the GPT-2 Text Generator API"}
