"""
merge.py: bake the LoRA adapter into the base weights and save a normal 16-bit HF model.

    source ~/.venvs/unsloth/bin/activate
    python merge.py                      # -> exports/merged-16bit/

Why: llama.cpp can apply a LoRA at runtime (--lora), but LM Studio and most hosts want one
self-contained model. Merging computes W + (alpha/r) * B @ A for every adapted layer once.
Unsloth merges shard by shard against the original 16-bit weights (not the 4-bit training
copy), so the result has no extra quantization loss and fits in laptop RAM.
"""
import argparse

from unsloth import FastLanguageModel

p = argparse.ArgumentParser()
p.add_argument("--adapter", default="outputs/vit-bot-lora/final")
p.add_argument("--out", default="exports/merged-16bit")
args = p.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(args.adapter, load_in_4bit=True)
model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {args.out}")
