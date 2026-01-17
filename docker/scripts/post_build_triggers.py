import os
import random
import time

from transformers import AutoTokenizer

MAX_RETRIES = 5


def download_tokenizer(model_name, save_path, token=None):
    os.makedirs(save_path, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            tokenizer.save_pretrained(save_path)
            return
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = random.uniform(0, min(60, 2 * (2**attempt)))
                print(f"Download failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
            else:
                print(f"Failed to download {model_name} after {MAX_RETRIES} attempts: {e}")
                print("Tokenizer will be downloaded at runtime if needed.")


HF_TOKEN_FILE = "/run/secrets/hf_token"

token = None
if os.path.exists(HF_TOKEN_FILE):
    with open(HF_TOKEN_FILE, "r") as f:
        token = f.read().strip()
    if token:
        print(f"Using HF token from secret file: {HF_TOKEN_FILE}")
if not token:
    token = os.getenv("HF_ACCESS_TOKEN")
    if token:
        print("Using HF token from HF_ACCESS_TOKEN environment variable")
    else:
        print("No HF token provided (some gated models may not be accessible)")

model_path = os.environ.get("MODEL_PREDOWNLOAD_PATH")

if os.getenv("DOWNLOAD_LLAMA_TOKENIZER") == "True":
    tokenizer_path = os.path.join(model_path, "llama-3.2-1b/tokenizer/")
    download_tokenizer("meta-llama/Llama-3.2-1B", tokenizer_path, token)
else:
    tokenizer_path = os.path.join(model_path, "e5-large-unsupervised/tokenizer/")
    download_tokenizer("intfloat/e5-large-unsupervised", tokenizer_path, token)
