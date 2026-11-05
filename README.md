# VIT Bot

A fine-tuned **Qwen3.5-4B** that speaks VIT (Vellore Institute of Technology) student culture, with five roast-y persona modes. Built end to end on a laptop: scrape → clean → distill a dataset → QLoRA fine-tune with Unsloth → GGUF → Telegram bot.

**Model:** [huggingface.co/Ajaythomas-crl/vit-bot-qwen3.5-4b](https://huggingface.co/Ajaythomas-crl/vit-bot-qwen3.5-4b)

| Mode | What it does |
|---|---|
| `/senior` | Tired, sarcastic senior answering a fresher, roast first, then one real tip |
| `/lawyer` | Dramatic attendance lawyer arguing your 73% case to the faculty advisor |
| `/excuse` | One straight-faced, VIT-accurate excuse you can actually say |
| `/ffcs` | The FFCS Oracle: reads your timetable like a horoscope |
| `/redtag` | Text game: get past Red Tag security without your ID card |

> **/lawyer** *73% in DSA, missed 4 classes sick, no medical cert. FA says debarred from FAT*
>
> Your FA, I present the landmark case of Vellore's First Sick Student vs. The Attendance Register, where the court ruled that illness without a cert counts as 'spiritually present' — the biometric scanner saw my spirit…

## Pipeline

```
r/Vit archive ──► clean + scrub PII ──► distill with a teacher LLM ──► QLoRA (Unsloth) ──► merge ──► GGUF Q4_K_M ──► llama.cpp / Telegram
 56.6k posts       50.2k posts           3,449 chat examples            RTX 4060, 40 min                 2.6 GB
 488k comments     top-5 replies/post    (5 modes)                       val loss 2.59 → 2.35
```

| Step | Script | What it does |
|---|---|---|
| 1. Collect | `scrape_vit.py` | Resumable downloader for the [Arctic Shift](https://arctic-shift.photon-reddit.com) Reddit archive; backs off on 422/429 and shrinks page size under load |
| 2. Clean | `clean.py` | Drops removed/bot/NSFW posts and reposts, scrubs usernames, emails, phones, VIT reg numbers and links, attaches each post's top 5 replies |
| 3. Explore | `explore.py` | Topic report from VIT-specific regexes plus TF-IDF → LSA → k-means clusters, used to pick the modes |
| 4. Distill | `make_dataset.py` | Real post titles become prompts; a teacher model (GLM-5.3 via OpenRouter, ~$4 total) writes persona answers using the top replies as private context. Paid-call budget cap, resumable, dedupe, safety and privacy re-checks |
| 5. Fine-tune | `train.py` | 4-bit QLoRA, r=16, loss on assistant turns only, 2 epochs (820 steps). `kaggle/` has a notebook version for free Kaggle GPUs |
| 6. Evaluate | `compare.py` | Same prompts through base vs fine-tuned (adapter on/off, same seed), scoring persona habits, length and VIT lore |
| 7. Export | `merge.py` + llama.cpp | Merge LoRA into 16-bit weights → `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` |
| 8. Serve | `bot.py`, `deploy/` | Telegram bot over any OpenAI-compatible server (llama-server / LM Studio); systemd units and a setup script for an Oracle Always Free ARM VM |

New to fine-tuning? [`LEARN.md`](LEARN.md) explains every step in plain English.

## Results

- **Validation loss** fell from 2.59 to 2.35 and never rose, so no overfitting. Epoch 2 added only ~0.025, so training stopped at 2 epochs.
- **Base vs fine-tuned:** the base model answers in generic chatbot style (markdown headers, "Pro Tip:", bullet lists). The fine-tune answers in one in-character paragraph with VIT references and ends with a real tip.
- **Speed:** ~14 tokens/s for the Q4_K_M GGUF on 12 laptop CPU threads; 4.5 GB VRAM while training.
- **Limitation:** it learned the style and common lore, but a 4B model still invents specifics (routes, rules, timings).

## Run it

```bash
# Chat with the published model (llama.cpp)
llama-cli -hf Ajaythomas-crl/vit-bot-qwen3.5-4b:Q4_K_M --jinja --reasoning-budget 0 \
  -sys "You are VIT Bot in /excuse mode: give one VIT-accurate excuse the student can actually say, delivered with a straight face. Under 80 words." \
  -p "Late by 15 minutes to an 8am lab in TT"

# Telegram bot against a local llama-server or LM Studio
pip install -r requirements-bot.txt
TELEGRAM_TOKEN=... LLM_URL=http://127.0.0.1:8080/v1 python bot.py
```

To reproduce training: `pip install unsloth`, build `data/train/*.jsonl` with steps 1–4 (needs an `LLM_API_KEY` for the teacher), then `python train.py`, then `python merge.py`.

## Privacy and safety

- The raw archive and generated dataset (`data/`) are **not** in this repo. Real comments were used only as private context for the teacher, never as training targets, and the bot is not trained to name real students or faculty.
- The Telegram bot sends a fixed Indian helpline reply (Tele-MANAS 14416) for suicide/self-harm messages instead of passing them to the model.
- Unofficial; not affiliated with VIT.
