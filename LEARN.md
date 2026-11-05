# VIT Bot: what we're building, explained from zero

## The goal in one sentence

Take a **small, free, open-source AI model**, **teach it to talk like a VIT senior** (five joke personas),
and **run it ourselves for ₹0/month** as a Telegram bot your friends can use.

---

## The big picture

```
 STEP 1          STEP 2            STEP 3              STEP 4            STEP 5          STEP 6
 Collect   ──►   Make the    ──►   Train the     ──►   Shrink it   ──►   Host it   ──►   Telegram
 r/Vit posts     lessons           small model         (GGUF)            (free VM)       bot
 (done)          (running now)     (next, on Linux)
```

Think of it like coaching a junior:

- **Step 1:** we read 50,000 r/Vit posts to learn what VITians actually ask.
- **Step 2:** we got an expert (a huge AI called GLM-5.3) to write 3,700 example answers in each persona's style.
- **Step 3:** a small AI studies those 3,700 examples until it can answer in the same style by itself.
- **Steps 4–6:** we compress that small AI so it runs on a cheap computer, and connect it to Telegram.

---

## The core ideas (read these once and the rest makes sense)

### 1. What is an LLM?
A **Large Language Model** is a program that has read a huge amount of text and learned to predict
"what word comes next". Ask it a question and it keeps predicting words until it has written an
answer. ChatGPT, Claude, Gemini, Qwen and Llama are all LLMs.

- **Big models** (hundreds of billions of "parameters", the numbers inside them) are smart but need
  huge servers. GLM-5.3 is one of these.
- **Small models** (1.5–4 billion parameters) are dumber in general, but run on a laptop or a cheap
  server. That's what our bot will be.

### 2. What is fine-tuning?
A small model out of the box knows English and general facts, but nothing about Q block lore or how a
tired 4th-year talks. **Fine-tuning = showing it thousands of examples of exactly the behaviour you
want, so it adjusts itself to copy that behaviour.**

It's like the difference between a smart fresher and a senior who has lived at VIT for 4 years: same
brain, different experience.

### 3. What is LoRA? (why this fits on your laptop)
A 3B model has 3 billion numbers. Changing all of them needs a lot of GPU memory. **LoRA (Low-Rank
Adaptation)** freezes the original model and trains a small "add-on" (~1% of the size) that nudges its
behaviour. It's like clipping a lens onto a camera instead of rebuilding the camera.

On top of that we load the frozen model in **4-bit** (each number stored with less precision), which
cuts memory about 4×. **4-bit + LoRA ("QLoRA")** is why an 8 GB laptop GPU can train a 3B model.
**Unsloth** is the library that makes this fast and memory-efficient.

### 4. What is distillation? (how we got training data)
To fine-tune, we need thousands of **(question → perfect answer)** pairs. Writing 3,700 funny
answers by hand would take weeks. So:

- **Teacher:** a huge, very good model (GLM-5.3) writes the answers. It cost about $2.
- **Student:** our small model learns from those answers.

That's **distillation**: the big model's skill gets "distilled" into the small one, for one narrow job.

### 5. Why not just use GLM / ChatGPT directly as the bot?
- **Cost:** every message would cost money, forever. Our small model costs ₹0 per message.
- **Control:** we own the model. No API going down, no rate limits, no terms changing.
- **Learning:** doing the full pipeline yourself is the actual industry skill.

### 6. What is a system prompt? (how one model has 5 personas)
Every conversation starts with a hidden instruction the user doesn't see, like:

> "You are VIT Bot in /lawyer mode: a dramatic attendance lawyer…"

We train **one** model where every example begins with one of 5 such instructions. It learns:
"when I see the /lawyer instruction, talk like the lawyer". The Telegram bot then just swaps the
instruction when someone types `/lawyer`. One model, five personalities.

### 7. Tokens, epochs, train/val
- **Token:** a chunk of text, about ¾ of a word. Models read and are billed in tokens.
- **Epoch:** one full pass over all 3,700 examples. We'll do 2–3. Too few and it doesn't learn the
  style; too many and it **overfits** (memorises the examples, gets worse at new questions).
- **Train / val split:** 95% of examples are used for learning (`train.jsonl`). 5% are hidden away
  (`val.jsonl`) to test it on questions it has never seen. Like keeping some past papers for a mock test.

### 8. What is GGUF / quantization? (step 4)
After training, the model is a few GB of numbers. **GGUF** is a file format for running models on
ordinary computers with **llama.cpp** / **Ollama**. **Q4_K_M** means "store each number in ~4 bits":
a 3B model becomes ~2 GB and runs on a CPU with little quality loss. That's what lets us host it on a
free server with no GPU.

---

## What we actually did, step by step

### Step 1: Collect the data (done)
| What | File | Why |
|---|---|---|
| Downloaded every r/Vit post since 2023 from the **Arctic Shift** archive (56,618 posts) | `scrape_vit.py` → `data/raw/posts.jsonl` | Reddit's own API is locked down and caps at ~1,000 posts. Arctic Shift is a public archive. |
| Downloading comments too (~450k, still running) | `data/raw/comments.jsonl` | Possible later use: facts for more accurate `/senior` tips. |
| Fixed the "422" crash | `scrape_vit.py` | The archive sometimes says "slow down". Now the script waits and retries instead of dying. |

### Step 1b: Clean it (done)
`clean.py`: removed bots, NSFW, deleted and duplicate posts, then **scrubbed private info**
(usernames → `<user>`, phone numbers, emails, VIT reg numbers like 21BCE1234, links). That left 50,196 posts.

### Step 1c: Find out what people talk about (done)
`explore.py` → `data/reports/topics.md`: counted topics (exams 17%, admissions 13%, hostel 11%,
placements 9%…) and found that rules/security and mess rants get the most upvotes. That's how we
picked the 5 modes:

| Mode | What it does | Why this one |
|---|---|---|
| `/senior` | Sarcastic 4th-year answers your question, with one real tip | Most posts are questions |
| `/lawyer` | Argues your attendance case with fake legal precedents | The "74%, debarred" cluster |
| `/excuse` | One VIT-accurate excuse | Cheap, pairs with `/lawyer` |
| `/ffcs` | FFCS Oracle: timetable horoscope | Huge FFCS and course-question volume |
| `/redtag` | Text game: get past security without your ID | Rules/security posts get the most upvotes |

### Step 2: Make the lessons (running now)
`make_dataset.py` does this for each example:

1. **Pick a real post** as the "seed". For `/senior` the real question *is* the user message. For the
   other modes the post just gives a situation, and the teacher invents the user's message.
2. **Add random "spice"**: a few VIT details, a texting style, and for redtag the tricks and outcome.
   The first pilot showed that without this the teacher repeats itself; every lawyer reply cited the
   same fake case.
3. **Ask GLM-5.3** (via OpenRouter) to write the reply as strict JSON.
4. **Check it automatically**: throw it away if it's too long, names a real person, contains a reg
   no or phone number, or touches a banned topic. Discards are logged with the reason.
5. **Save it**, then at the end shuffle everything into `data/train/train.jsonl` and `val.jsonl`.

What one finished training example looks like:
```json
{"messages": [
  {"role": "system",    "content": "You are VIT Bot in /senior mode: a tired, sarcastic VIT senior..."},
  {"role": "user",      "content": "Feeling lonely all day ! No friends to talk to ."},
  {"role": "assistant", "content": "VIT loneliness is basically a slot everyone registers for and nobody drops... Real tip: sit with random batchmates at Foodys..."}
]}
```

**Why real comments are NOT the answers:** they're real people's words (privacy), they vary a lot in
quality, and they don't have one consistent persona voice. Real posts give the *questions*; the teacher
writes the *answers*.

**Safety rules we chose** (later cut to the minimum: only suicide/self-harm, sexual violence and caste/religion jokes are still blocked; everything else, including SOFT topics in every mode, is allowed):
- **Excluded completely (`HARD`):** suicide/self-harm, death/grief, harassment/ragging, serious
  illness or accidents, protests/police, caste/religion/region, sexual content. Roasting those
  would be cruel, not funny.
- **Kept, answered rude (`SOFT`):** stress, loneliness, homesickness, crushes, breakups, mental
  health. They're roasted, but the joke is aimed at the situation and VIT, never at the person's
  feelings, and always ends with a real tip.
- The Telegram bot will have a hard-coded catch: messages about suicide/self-harm get a fixed
  helpline reply (Tele-MANAS 14416) and never reach the model.

**Pilots before the full run:** we ran 10 examples per mode twice (~$0.03 each), read them, fixed
the prompts, then did the full run (~3,700 examples, ~$2). Always pilot before you spend.

---

## What's next (on Linux)

### Step 3: Train (on your RTX 4060, 8 GB)
1. Install the NVIDIA driver + PyTorch + Unsloth.
2. Pick a base model: a current 1.5B–4B *instruct* model (Qwen / Gemma / Llama family). "Instruct"
   means it already knows how to chat, so we only teach it the VIT style.
3. Run training: about **35–60 min** for 3B, 3 epochs. Watch the **loss** number go down (loss means
   "how wrong it is on the examples").
4. **Evaluate:** ask the base model and our fine-tuned model the same held-out questions, side by side.
   Is it funnier? More VIT? Does it stay in persona? Does it ever name real people?

### Step 4: Export
Merge the LoRA add-on into the model, convert to **GGUF Q4_K_M** (~2 GB), test it locally with llama.cpp.

### Step 5: Host
Run it with llama.cpp server or Ollama on **Oracle Cloud Always Free** (4 ARM cores, 24 GB RAM, ₹0).

### Step 6: Telegram bot
`python-telegram-bot`: `/senior`, `/lawyer`, etc. switch the system prompt. Add a per-user rate limit
and the safety guard. Share in club groups.

---

## Glossary

| Term | Meaning |
|---|---|
| **Parameters** | The numbers inside a model. "3B" = 3 billion. More usually means smarter but heavier. |
| **Base vs instruct model** | Base = raw text predictor. Instruct = already trained to chat and follow instructions. We start from instruct. |
| **Fine-tuning** | Further training on your own examples to change behaviour. |
| **LoRA / QLoRA** | Train a tiny add-on instead of the whole model / do that on a 4-bit model. Fits on 8 GB. |
| **Unsloth** | Library that makes LoRA training ~2× faster with less memory. |
| **Distillation** | Big "teacher" model writes examples, small "student" model learns from them. |
| **System prompt** | Hidden instruction at the start of a chat that sets the persona. |
| **Token** | ~¾ of a word. Unit of text for models and billing. |
| **Epoch** | One pass over the whole training set. |
| **Loss** | How wrong the model is on training examples. Should go down. |
| **Overfitting** | Memorising examples instead of learning the style. Caught with the val set. |
| **Train/val split** | Learn on most examples, test on a few it never saw. |
| **Quantization** | Storing numbers with fewer bits (16 → 4) to shrink the model. |
| **GGUF** | File format for running quantized models on normal computers (llama.cpp, Ollama). |
| **llama.cpp / Ollama** | Programs that run GGUF models, on CPU or GPU. |
| **OpenRouter** | One API key that gives access to hundreds of models. We used it for GLM-5.3. |
| **JSONL** | Text file with one JSON object per line. All our data files use it. |

## Cost and time so far
- Money: ~$2 OpenRouter (one-off). Everything else is free.
- Data: 56,618 posts, ~450k comments, 37.6k usable seeds, ~3,700 training examples.
- Files to know: `HANDOFF.md` (for Claude), `README.md` (how to run), this file (to understand).
