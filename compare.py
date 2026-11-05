"""
compare.py: base vs fine-tuned replies on held-out val prompts, for one LoRA adapter.

    source ~/.venvs/unsloth/bin/activate
    python compare.py --adapter outputs/vit-bot-lora/final --label local-qwen3.5 --out outputs/cmp_local.jsonl
    python compare.py --report outputs/cmp_local.jsonl outputs/cmp_kaggle.jsonl   # -> outputs/compare.md

The same loaded model answers twice per prompt: once with the LoRA switched off (base) and once
with it on (fine-tuned), using the same sampling seed, so the only difference is the adapter.
"""
import argparse
import json
import re
from collections import defaultdict

MAX_WORDS = {"ffcs": 140}  # everything else 120, same as make_dataset.py
BAND = {"senior": (40, 85), "lawyer": (50, 90), "excuse": (20, 60), "ffcs": (45, 85), "redtag": (8, 45)}
LORE = re.compile(r"\b(ffcs|vtop|slots?|cat ?[12i]+|fat|da|75%|debar\w*|proxy|od|faculty advisor|fa|"
                  r"mb|sjt|tt|gdn|smv|foodys|greenos|limra|night canteen|mess|hostel|curfew|in-?time|"
                  r"biometric|red ?tag|id card|gravitas|riviera|volsbb|katpadi|vellore|anna|warden|"
                  r"cgpa|gpa|add.?drop|re-?fat|placements?|bhai|machan)\b", re.I)
ASSISTANTY = re.compile(r"as an ai|i'm sorry|i cannot|i can't help|options:|\*\*|^#", re.I | re.M)


def mode_of(system):
    return re.search(r"/(\w+) mode", system).group(1)


def strip_think(t):
    return re.sub(r"<think>.*?</think>\s*", "", t, flags=re.S).replace("<think>", "").strip()


def pick_prompts(path, per_mode):
    """First `per_mode` val examples per mode. Redtag keeps two user turns to test reacting mid-game."""
    picked = defaultdict(list)
    for line in open(path):
        msgs = json.loads(line)["messages"]
        mode = mode_of(msgs[0]["content"])
        if len(picked[mode]) >= per_mode:
            continue
        if mode == "redtag" and len(msgs) >= 5:
            prompt, ref = msgs[:4], msgs[4]["content"]
        else:
            prompt, ref = msgs[:2], msgs[2]["content"]
        picked[mode].append((prompt, ref))
    return picked


def generate(args):
    import torch
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=1024, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)
    FastLanguageModel.for_inference(model)

    # Stop at the end of the assistant turn. Qwen3.5 ships no generation_config, so without this
    # it keeps going and writes the user's next /redtag move itself.
    stop_ids = list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})

    def reply(msgs, seed):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Training replies all started with an empty think block; pre-fill it so the model doesn't have
        # to generate the (barely trained) </think> token itself.
        if not text.rstrip().endswith("</think>"):
            text += "<think>\n\n</think>\n\n"
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.manual_seed(seed)
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.9,
                             eos_token_id=stop_ids)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    with open(args.out, "w") as f:
        for mode, items in pick_prompts(args.val, args.per_mode).items():
            for i, (prompt, ref) in enumerate(items):
                with model.disable_adapter():
                    base = reply(prompt, seed=i)
                ft = reply(prompt, seed=i)
                rec = {"label": args.label, "mode": mode, "prompt": prompt, "reference": ref, "base": base, "ft": ft}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{mode} {i}] FT: {strip_think(ft)[:150]!r}", flush=True)


def stats(texts, mode):
    words = [len(strip_think(t).split()) for t in texts]
    lo, hi = BAND[mode]
    return {
        "avg words": sum(words) / len(words),
        "under limit": sum(w <= MAX_WORDS.get(mode, 120) for w in words) / len(words),
        "in target band": sum(lo <= w <= hi for w in words) / len(words),
        "VIT terms/reply": sum(len(set(m.lower() for m in LORE.findall(strip_think(t)))) for t in texts) / len(texts),
        "assistant-y": sum(bool(ASSISTANTY.search(strip_think(t))) for t in texts) / len(texts),
        "think tags shown": sum("think>" in t for t in texts) / len(texts),
    }


def report(paths, out_md):
    runs = [[json.loads(l) for l in open(p)] for p in paths]
    md = ["# VIT Bot: base vs fine-tuned\n"]
    md.append("Numbers are averages over held-out val prompts. *in target band* = the word range the "
              "teacher was asked for; *assistant-y* = generic-chatbot tells (markdown, menus, refusals).\n")
    modes = list(dict.fromkeys(r["mode"] for r in runs[0]))
    for mode in modes:
        md.append(f"\n## /{mode}\n")
        cols = []
        for run in runs:
            rs = [r for r in run if r["mode"] == mode]
            lab = rs[0]["label"]
            cols += [(f"{lab} base", stats([r["base"] for r in rs], mode)),
                     (f"{lab} FT", stats([r["ft"] for r in rs], mode))]
        cols.append(("reference", stats([r["reference"] for r in rs], mode)))
        md.append("| metric | " + " | ".join(c[0] for c in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for k in cols[0][1]:
            fmt = (lambda v: f"{v:.0f}") if k == "avg words" else (lambda v: f"{v:.1f}") if "/" in k else (lambda v: f"{v:.0%}")
            md.append(f"| {k} | " + " | ".join(fmt(c[1][k]) for c in cols) + " |")
        for j in range(min(2, len(runs[0]))):
            ex = [r for r in runs[0] if r["mode"] == mode][j]
            md.append(f"\n**Prompt:** {ex['prompt'][-1]['content'][:400]}\n")
            for run in runs:
                r = [r for r in run if r["mode"] == mode][j]
                md.append(f"- **{r['label']} base:** {strip_think(r['base'])[:600]}")
                md.append(f"- **{r['label']} FT:** {strip_think(r['ft'])[:600]}")
            md.append(f"- *reference (teacher):* {ex['reference'][:600]}")
    open(out_md, "w").write("\n".join(md) + "\n")
    print(f"Wrote {out_md}")


p = argparse.ArgumentParser()
p.add_argument("--adapter")
p.add_argument("--label")
p.add_argument("--out")
p.add_argument("--val", default="data/train/val.jsonl")
p.add_argument("--per-mode", type=int, default=6)
p.add_argument("--max-new", type=int, default=220)
p.add_argument("--report", nargs="+")
p.add_argument("--report-out", default="outputs/compare.md")
a = p.parse_args()
if a.report:
    report(a.report, a.report_out)
else:
    generate(a)
