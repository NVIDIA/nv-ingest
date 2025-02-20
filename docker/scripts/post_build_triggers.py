import os
from transformers import AutoTokenizer

if os.getenv("DOWNLOAD_LLAMA_TOKENIZER") == "True":
    tokenizer_path = "/workspace/models/tokenizer/"
    os.makedirs(tokenizer_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=os.getenv("HF_ACCESS_TOKEN"))
    tokenizer.save_pretrained(tokenizer_path)
