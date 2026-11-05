#!/usr/bin/env bash
# setup_vm.sh: one-time setup of VIT Bot on the Oracle Always Free ARM VM (Ubuntu 24.04).
#
# From the laptop (project folder), copy the files over first (~41 MB total):
#   ssh ubuntu@<IP> mkdir -p vitbot/models
#   scp bot.py requirements-bot.txt deploy/* ubuntu@<IP>:vitbot/
#   scp exports/vit-bot-lora-f16.gguf ubuntu@<IP>:vitbot/models/
# Then on the VM:
#   bash ~/vitbot/setup_vm.sh
#   nano ~/vitbot/.env          # TELEGRAM_TOKEN=... from @BotFather
#   sudo systemctl restart vit-bot
set -euo pipefail

LLAMA_TAG=b12128   # same llama.cpp release used to export the GGUFs on the laptop
APP="$HOME/vitbot"
cd "$APP"

echo "== packages"
sudo apt-get update -q
sudo apt-get install -y -q python3-venv curl

echo "== llama.cpp $LLAMA_TAG (prebuilt ARM64)"
if [ ! -x "$APP/llama/llama-server" ]; then
  curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-arm64.tar.gz" | tar -xz -C "$APP"
  ln -sfn "$APP/llama-$LLAMA_TAG" "$APP/llama"
fi

echo "== python venv"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q -r requirements-bot.txt huggingface_hub

echo "== base model (2.7 GB, straight from Hugging Face)"
"$APP/.venv/bin/hf" download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir "$APP/models"
test -f "$APP/models/vit-bot-lora-f16.gguf" || { echo "Missing models/vit-bot-lora-f16.gguf: scp it from the laptop"; exit 1; }

echo "== .env"
if [ ! -f "$APP/.env" ]; then
  printf 'TELEGRAM_TOKEN=\nLLM_URL=http://127.0.0.1:8080/v1\nMAX_PARALLEL=1\n' > "$APP/.env"
  chmod 600 "$APP/.env"
fi

echo "== systemd services"
sudo cp "$APP/llama-server.service" "$APP/vit-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl enable vit-bot
grep -q '^TELEGRAM_TOKEN=.\+' "$APP/.env" && sudo systemctl restart vit-bot \
  || echo "Put your token in $APP/.env, then: sudo systemctl restart vit-bot"

echo "Done. Logs: journalctl -u llama-server -f   /   journalctl -u vit-bot -f"
