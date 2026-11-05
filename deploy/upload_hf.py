"""
upload_hf.py: publish VIT Bot to the Hugging Face Hub.

    HF_TOKEN=... python deploy/upload_hf.py [--repo Ajaythomas-crl/vit-bot-qwen3.5-4b]

Uploads the model card, the merged Q4_K_M GGUF, the LoRA GGUF and the PEFT adapter.
Never uploads data/ (the dataset stays private).
"""
import argparse

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--repo", default="Ajaythomas-crl/vit-bot-qwen3.5-4b")
args = p.parse_args()

api = HfApi()
api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
api.upload_file(path_or_fileobj="exports/hf/README.md", path_in_repo="README.md", repo_id=args.repo,
                commit_message="Model card")
api.upload_folder(folder_path="outputs/vit-bot-lora/final", path_in_repo="lora", repo_id=args.repo,
                  commit_message="PEFT LoRA adapter")
api.upload_file(path_or_fileobj="exports/vit-bot-lora-f16.gguf", path_in_repo="vit-bot-lora-f16.gguf",
                repo_id=args.repo, commit_message="LoRA adapter (GGUF)")
print("Uploading the 2.6 GB merged GGUF...")
api.upload_file(path_or_fileobj="exports/vit-bot-qwen3.5-4b-Q4_K_M.gguf",
                path_in_repo="vit-bot-qwen3.5-4b-Q4_K_M.gguf", repo_id=args.repo,
                commit_message="Merged model, Q4_K_M GGUF")
print(f"Done: https://huggingface.co/{args.repo}")
