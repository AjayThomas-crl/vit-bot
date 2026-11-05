"""
modal_merge.py: merge the VIT Bot LoRA into Qwen3.5-4B on Modal and save it to a Modal Volume.

    modal run deploy/modal_merge.py

Nothing is uploaded from the laptop: the container downloads the public base model and the
85 MB LoRA adapter straight from Hugging Face, merges them in bf16 on CPU, and writes the
full safetensors model to the `vit-bot-weights` Volume at /merged. A Modal Dedicated Endpoint
then serves it:

    modal endpoint create --name vit-bot --model Qwen/Qwen3.5-4B \
      --custom-volume-name vit-bot-weights --custom-volume-path /merged
"""
import modal

BASE = "unsloth/Qwen3.5-4B"          # same weights as Qwen/Qwen3.5-4B; the adapter was trained on it
ADAPTER = "Ajaythomas-crl/vit-bot-qwen3.5-4b"
OUT = "/vol/merged"

app = modal.App("vit-bot-merge")
vol = modal.Volume.from_name("vit-bot-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.17.0", "peft==0.21.2", "accelerate", "huggingface_hub", "safetensors")
)


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3600)
def merge():
    import os
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base_dir = snapshot_download(BASE)
    model = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16)

    # Sanity check later: an adapted weight must change after merging.
    probe = "model.language_model.layers.0.mlp.down_proj.weight"
    before = dict(model.named_parameters())[probe].detach().clone()

    model = PeftModel.from_pretrained(model, ADAPTER, subfolder="lora")
    model = model.merge_and_unload()
    after = dict(model.named_parameters())[probe].detach()
    delta = (after.float() - before.float()).norm().item()
    print(f"LoRA delta on {probe}: {delta:.4f}")
    assert delta > 0, "merge changed nothing: adapter did not attach"

    shutil.rmtree(OUT, ignore_errors=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")

    # Tokenizer, chat template and processor configs come straight from the base repo.
    for name in os.listdir(base_dir):
        if not name.endswith(".safetensors") and name not in ("model.safetensors.index.json", "config.json"):
            src = os.path.join(base_dir, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(OUT, name))

    vol.commit()
    for name in sorted(os.listdir(OUT)):
        print(f"{name:50} {os.path.getsize(os.path.join(OUT, name)) / 1e6:9.1f} MB")


@app.local_entrypoint()
def main():
    merge.remote()
