"""
modal_llama.py: serve VIT Bot on Modal with llama.cpp on a T4 GPU, scaling to zero when idle.

    modal secret create vit-bot-llm LLM_API_KEY=<random string>   # once
    modal deploy deploy/modal_llama.py

Exposes llama-server's OpenAI-compatible API (/v1/chat/completions) at the URL Modal prints.
Requests need `Authorization: Bearer <LLM_API_KEY>`, so strangers can't spend the credits.
The 2.6 GB Q4_K_M GGUF is pulled from the Hugging Face repo at image build time and baked
into the image, so cold starts don't re-download it. At most one container runs, and it shuts
down after 5 idle minutes.
"""
import os
import subprocess

import modal

LLAMA_TAG = "b12128"
REPO = "Ajaythomas-crl/vit-bot-qwen3.5-4b"
GGUF = "vit-bot-qwen3.5-4b-Q4_K_M.gguf"
REL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "libgomp1")
    .run_commands(
        "mkdir -p /opt/llama",
        f"curl -fsSL {REL}/llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C /opt/llama",
        f"curl -fsSL {REL}/cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C /opt/llama",
        # Put every bundled shared library on the loader path and expose the server binary.
        "find /opt/llama -name '*.so*' -printf '%h\\n' | sort -u > /etc/ld.so.conf.d/llama.conf && ldconfig",
        "ln -sf $(find /opt/llama -name llama-server -type f | head -1) /usr/local/bin/llama-server",
    )
    .pip_install("huggingface_hub")
    .run_commands(f"hf download {REPO} {GGUF} --local-dir /models")
)

app = modal.App("vit-bot")


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("vit-bot-llm")],
    max_containers=1,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=8080, startup_timeout=300)
def serve():
    subprocess.Popen([
        "llama-server",
        "-m", f"/models/{GGUF}",
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99",                 # whole model on the GPU
        "-c", "8192", "-np", "4",     # 4 parallel chats, 2048 tokens each
        "--jinja", "--reasoning-budget", "0",
        "--api-key", os.environ["LLM_API_KEY"],
    ])
