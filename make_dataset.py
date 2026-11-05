"""
Build the VIT Bot training set by distillation: real r/Vit posts are the seeds,
a big "teacher" model writes the persona answers, and the small model learns from those.

Usage:
    python make_dataset.py --dry-run                  # no API calls: pool sizes + sample prompts
    python make_dataset.py --pilot 10                 # 10 examples per mode, so you can check quality first
    python make_dataset.py                            # full run (resumable, Ctrl+C safe)
    python make_dataset.py --build                    # only rebuild train/val from what's generated

Teacher model: any OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter, Together,
a local Ollama, ...). Set three env vars:
    LLM_BASE_URL   e.g. https://api.groq.com/openai/v1
    LLM_API_KEY    your key (never commit it)
    LLM_MODEL      the model name on that provider

Output:
    data/gen/<mode>.jsonl           raw teacher outputs, one per seed post (resume state)
    data/train/train.jsonl, val.jsonl   chat format: {"messages": [system, user, assistant, ...]}

Why it's built this way:
  * Seeds, not copies. We take real questions/situations, but every answer is freshly
    written by the teacher, so the bot never quotes or names real people.
  * One model, five modes. Each mode has its own short system prompt; the small model
    learns to switch persona on it. Telegram commands just pick the system prompt.
  * The val split is by seed post, so no val example's seed was seen in training.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

import requests

CLEAN_PATH = os.path.join("data", "clean", "posts.jsonl")
GEN_DIR = os.path.join("data", "gen")
TRAIN_DIR = os.path.join("data", "train")
VAL_FRACTION = 0.05
MAX_WORDS = 120          # Telegram-friendly; per assistant turn
# The oracle has to do a whole mystic format AND fit in the seniors' facts; at 120 a quarter of
# comment-backed ffcs answers were thrown away at 121-130 words.
MAX_WORDS_MODE = {"ffcs": 140}
SEED = 42

# --------------------------------------------------------------------------
# What the *small* model sees at train + inference time. Keep these short:
# they're repeated in every example and every Telegram request.
# --------------------------------------------------------------------------
BOT_SYSTEM = {
    "senior": "You are VIT Bot in /senior mode: a tired, sarcastic VIT Vellore senior answering a "
              "fresher. Roast lightly, then end with one genuinely useful tip. Under 100 words.",
    "lawyer": "You are VIT Bot in /lawyer mode: a dramatic attendance lawyer arguing the student's "
              "case to their faculty advisor, citing absurd 'precedents' and VIT rules. Under 100 words.",
    "excuse": "You are VIT Bot in /excuse mode: give one VIT-accurate excuse the student can "
              "actually say, delivered with a straight face. Under 80 words.",
    "ffcs":   "You are VIT Bot in /ffcs mode: the FFCS Oracle. Read the student's timetable, slots or "
              "course plan like a horoscope: dramatic prophecy, one real tip. Under 100 words.",
    "redtag": "You are VIT Bot in /redtag mode: a text game. You are a Red Tag security anna at the "
              "VIT gate and the player has no ID card. Stay in character, react to their moves, keep "
              "replies short, and end the game with a clear win or loss after a few turns.",
}

# --------------------------------------------------------------------------
# What the *teacher* gets. Rich context here costs nothing at inference time.
# --------------------------------------------------------------------------
LORE = """VIT Vellore lore you can use: FFCS (Fully Flexible Credit System; slot-based course and
faculty registration, servers crash, everyone fights for good faculty), VTOP (student portal, always
down when needed), slots (A1, B2, TA1, L-slots for labs), CAT-1, CAT-2, FAT (final assessment test),
DA (digital assignments), 75% attendance rule, debarment, proxy, OD (on duty), medical leave,
faculty advisor (FA), proctor, HOD, blocks (MB, SJT, TT, GDN, SMV, PRP), men's hostel blocks
(A to T, Q block memes) and ladies' hostel, mess (veg, non-veg, special mess), Foodys, Greenos,
Limra, night canteen, hostel in-time/curfew, biometric, Red Tag security ("ID pehno!"), ID card
rules, dress code, Gravitas (tech fest), Riviera (cultural fest), VOLSBB wifi, Katpadi station,
Vellore heat (45 degrees), relative grading, CGPA, 9-pointers, placements (dream, super dream),
NPTEL, summer/weekend semester, Add/Drop, lab coats, the long MB to TT walk."""

RULES = """Hard rules:
- Funny and roast-y, never cruel. Punch up at the system (VTOP, FFCS, rules, heat), not at people.
- Never name or describe real people: no professors, students, staff, or reddit users. Use generic
  roles ("your FA", "that one prof", "the warden").
- No jokes about caste, religion, region/state, gender or body.
- Stress, loneliness, homesickness, breakups, mental health: still rude and sarcastic, but roast the
  situation and VIT, never mock the person's feelings, and the closing tip must be genuinely useful
  (e.g. friends, the campus counselling centre, their proctor).
- Light Hinglish/Tanglish sprinkles are fine; mostly English.
- No links, no phone numbers, no reg numbers, no emoji spam (one max, often none).
- Never invent official facts: no made-up fees, rule numbers, CGPA cutoffs or dates. Jokes can be absurd;
  advice must be safe and general when unsure ("check VTOP", "ask your FA/proctor").
- Be original every time. Do NOT open with "Ah", "Ah yes", "Beta", "Bro", "Listen", "Your Honour, my
  client stands accused", "The Oracle has seen...". Avoid stock phrases; find a fresh first line.
- Don't fall back on "VTOP is down" or "the MB to TT walk" as the punchline unless the spice lists them.
- Output strict JSON only, nothing else. Escape any double quotes inside strings."""

# Pilot showed the teacher copies whatever examples the prompt gives (every lawyer reply cited the
# same fake case). So instead of fixed examples, each call gets a random "spice": a few lore items
# to lean on, a user style, and for /redtag the tricks and the outcome. More variety -> a small
# model that doesn't repeat itself.
LORE_ITEMS = [
    "FFCS registration day", "VTOP being down", "slot clashes", "Add/Drop week", "CAT-1", "CAT-2",
    "FAT", "DA deadlines", "relative grading", "the 75% rule", "debarment", "proxy attempts",
    "OD forms", "medical leave", "biometric", "faculty advisor", "proctor", "HOD cabin",
    "MB", "SJT", "TT", "GDN", "SMV", "PRP", "the long walk between blocks", "lifts in SJT",
    "mess food", "special mess", "Foodys", "Greenos", "Limra", "night canteen", "hostel in-time",
    "warden", "laundry", "room allotment", "Red Tag security", "ID card rule", "dress code",
    "Gravitas", "Riviera", "clubs and chapters", "VOLSBB wifi", "Katpadi station", "autos to Katpadi",
    "Vellore heat", "power cuts", "lab coats", "lab FAT", "NPTEL", "summer semester", "weekend semester",
    "placements", "internships", "CGPA panic", "9-pointers", "xerox shop queue", "rain flooding the roads",
]
USER_STYLES = [
    "panicked fresher, lots of caps", "chill 3rd year, lowercase, no punctuation",
    "Hinglish texter", "a bit of Tanglish", "dramatic, overthinks everything", "short and blunt",
    "polite and formal", "day scholar", "hosteller", "lateral/2nd year who just joined a club",
    "final year, placement stressed", "sleep deprived, typos",
]
REDTAG_TRICKS = [
    "claims to be a visiting parent", "says they're a delivery person", "hides in a crowd of club members",
    "pretends to be a professor", "claims they're the new mess supervisor","shows a photo of their ID on phone",
    "uses a friend's ID", "argues rules philosophically", "speaks only in Tamil to confuse", "flatters the anna",
    "claims to be a Riviera performer", "says the lab is on fire and they must help", "offers to fix the anna's phone",
    "tries the back gate", "brings a fake permission letter", "pretends to be on a video call with the Dean",
    "challenges the anna to a quiz", "acts like a lost tourist", "cries", "speedwalks past during shift change",
]
ORACLE_FORMATS = [
    "a single invented tarot card (make up its name)", "a weather forecast", "a zodiac horoscope",
    "a palm reading", "an ancient prophecy scroll", "a cricket commentary of the semester",
    "a fortune cookie followed by an explanation", "a cosmic stock market report", "a train announcement at Katpadi",
    "a GPS navigation voice", "a nature documentary narrator", "a Bigg Boss style announcement",
]
REDTAG_OUTCOMES = ["WIN", "CAUGHT", "CAUGHT", "WIN", "CAUGHT", "DRAW: the anna lets them in but extracts a funny price"]

LENGTH = {"senior": "40-85 words", "lawyer": "50-90 words", "excuse": "20-60 words",
          "ffcs": "45-85 words", "redtag": "8-45 words per assistant turn"}

TEACHER_TASK = {
    "senior": """Write the /senior reply to the student's question below.
Persona: tired, sarcastic VIT senior, 4th year, has seen it all. Light roast first, then one genuinely
useful tip at the end. The user message is the real question; don't rewrite it.
Return JSON: {"assistant": "..."}""",
    "lawyer": """The real post below is just a situation seed. Turn it into an attendance or academic case.
1) Write the student's message asking the lawyer to defend them (1-2 sentences, Telegram style, in the
   user style given). Use the seed's own details: subject, situation and numbers should come from it.
2) Write the /lawyer reply: a dramatic plea to the faculty advisor with ONE invented, ridiculous legal
   precedent whose name comes from THIS specific situation, plus VIT rules twisted creatively.
Return JSON: {"user": "...", "assistant": "..."}""",
    "excuse": """The real post below is just a situation seed.
1) Write the student's message asking for an excuse (one sentence, in the user style given), about
   something that fits the seed.
2) Write the /excuse reply: ONE excuse, VIT-specific, built from the seed and the spice, plausible
   enough to say out loud. Short and punchy, no bonus lists.
Return JSON: {"user": "...", "assistant": "..."}""",
    "ffcs": """The real post below is just a situation seed.
1) Write the student's message about their FFCS / timetable / courses / semester plan (in the user style
   given), with specifics that come from the seed, not a generic "8am slots" complaint.
2) Write the /ffcs Oracle reply: a prophecy about their semester in the mystic format given in the
   spice, with one genuinely useful, safe FFCS or academics tip hidden in it.
Return JSON: {"user": "...", "assistant": "..."}""",
    "redtag": """The real post below is just a flavour seed (a rule, a place, a vibe).
Write a short text-game conversation. The bot is a Red Tag security anna at a VIT gate; the player has
no ID card and tries to get in. 3 to 5 exchanges. The first user turn is "/redtag", "start" or a first
move. The anna stays in character (catchphrases welcome but vary them), reacts to each move, and the
final assistant turn clearly ends the game with the outcome given below and a one-line result.
Return JSON: {"turns": [{"user": "...", "assistant": "..."}, ...]}""",
}


# --------------------------------------------------------------------------
# Real replies as private fact notes (--use-comments)
# --------------------------------------------------------------------------
# GLM knows VIT in general but guesses the specifics. The top replies under a post usually hold
# the real answer from people who've been through it. We show them to the teacher as notes so the
# tip is CORRECT, while the bot itself never sees, quotes or names anyone from them.
USE_COMMENTS = False
COMMENT_MODES = {"senior", "ffcs", "lawyer", "excuse", "redtag"}
FACT_MODES = {"senior", "ffcs"}             # replies = the answer's substance; elsewhere = creative raw material
CREATIVE_NOTES_HEADER = """

--- what real VIT students replied under this post ---
Use these as RAW MATERIAL for the comedy: real situations, real excuses people used, what faculty or
security actually did, real VIT slang and the actual humour of the place. Your reply should feel like it
came from someone who lives at VIT, built on these details instead of generic jokes.
- Rework them in the persona's own voice. Never copy a sentence word for word, never quote, never name
  or describe anyone, never say "a comment said".
- Ignore replies that are off-topic or just "same bro". If nothing is usable, rely on the seed and spice.
"""
NOTES_HEADER = """

--- what real VIT seniors replied under this post ---
BUILD THE ANSWER FROM THESE. The substance of your reply (the advice, facts, process, warnings, what
actually worked) must come from what these seniors said, not from your own general knowledge. Keep
the persona's roast around it, but the useful content is theirs.
- Paraphrase in the persona's own voice. Never copy a sentence word for word, never quote, never name
  or describe anyone, never say "a comment said". Phrases like "seniors who've been through this say..."
  are fine.
- Prefer the most upvoted, sensible consensus. Skip replies that are jokes, wrong or just "same bro".
- If the replies hold nothing useful, fall back to safe general advice.
- Length: 50-100 words for this one, so the real advice fits.
"""


def senior_notes(p, k=5, min_score=2, max_chars=400):
    notes = []
    for c in p.get("top_comments", []):          # already sorted by score, PII-scrubbed by clean.py
        body = re.sub(r"\s+", " ", c["body"]).strip()
        if c["score"] < min_score or len(body) < 15 or HARD.search(body) or PEOPLE.search(body):
            continue
        notes.append(f"({c['score']} upvotes) {body[:max_chars]}")
        if len(notes) == k:
            break
    return notes


def spice(mode, rng):
    lines = [f"Lean on these VIT details (use 1-3, not all): {', '.join(rng.sample(LORE_ITEMS, 5))}.",
             f"Length: {LENGTH[mode]}."]
    if mode != "senior":                      # senior's user message is the real post
        lines.insert(1, f"User style: {rng.choice(USER_STYLES)}.")
    if mode == "ffcs":
        lines.append(f"Mystic format: {rng.choice(ORACLE_FORMATS)}.")
    if mode == "redtag":
        tricks = rng.sample(REDTAG_TRICKS, rng.randint(2, 4))
        # The pilot showed the teacher drifting back to its favourites (Dean video call, visiting parent),
        # so pin it to exactly these tricks.
        lines.append(f"The player's tricks, in order: {'; '.join(tricks)}. Use ONLY these {len(tricks)} tricks "
                     f"(one per player turn after the opening), no others.")
        lines.append(f"Outcome: {rng.choice(REDTAG_OUTCOMES)}.")
    return "\n".join(lines)

# --------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------
# HARD: never trained on, in any mode. Kept narrow on purpose.
#   Suicide/self-harm is here not because it's taboo but because the bot shouldn't improvise on it:
#   the Telegram layer will answer those messages with a fixed helpline reply instead.
HARD = re.compile(
    # Ajay asked for no safety filtering beyond the bare minimum (2026-11-05). What's left is only what a
    # rude bot must never improvise on: suicide/self-harm, sexual violence and leaked nudes, and
    # caste/religion/region identity jokes. Death, accidents, police, protests, ragging etc. are now allowed.
    r"\b(suicid|su.?cide|s\*\*cide|unalive|kms\b|self.?harm|kill(?:ed)? (?:my|him|her|them)sel(?:f|ves)|end (?:my|it) all|"
    r"took (?:his|her|their|my) (?:own )?life|hang(?:ed|ing) (?:him|her)self|"
    r"rape|molest|sexual(?:ly)? (?:assault|harass)|leaked|nude|"
    r"caste|religio|hindu|muslim|christian|north ?indian|south ?indian|northies?\b|southies?\b)",
    re.I)
# SOFT: real student life, worth keeping, but not roast material. Allowed only in /senior, where the
# teacher is told to drop the sarcasm and be a kind senior. That teaches the bot when NOT to joke.
SOFT = re.compile(
    r"\b(depress|anxiety|anxious|lonely|loneliness|homesick|stress|burn.?out|panic attack|therapy|therapist|"
    r"mental health|counsell?or|overwhelm|cry(?:ing)?\b|sad\b|hopeless|"
    r"dating|girlfriend|boyfriend|crush|break.?up|relationship|parents? (?:fight|pressure))",
    re.I)
SENSITIVE = HARD   # seeds: only HARD blocks a post outright
# Posts that would teach the bot to talk about specific real people.
PEOPLE = re.compile(r"\b(viswanathan|chancellor|faculty review|review (pls|please)|reviews? of|dr\.?\s|prof\.?\s|sir\s+[A-Z]|"
                    r"ma'?am\s+[A-Z]|<user>)", re.I)
SKIP_FLAIRS = {"buy / sell / trade", "faculty review", "ffcs faculty review",
               "lost & found", "mod post", "captures 📷", "ama"}   # "Serious"/"Confessions" kept: HARD/SOFT handle them

TOPIC = {
    "attendance": re.compile(r"\battendance\b|\bdebar|\b7[0-4](\.\d+)? ?%|\bproxy\b|\bbunk|\bod\b|"
                             r"\bon.?duty\b|\bmedical leave\b|\bbiometric\b|\bclasses?\b|\babsent", re.I),
    "exams": re.compile(r"\bcat[- ]?(?:1|2|i{1,2})\b|\bfat\b|\bexams?\b|\bda\b|\bassignment|\bdeadline|"
                        r"\blab\b|\bquiz|\bsubmit", re.I),
    "ffcs": re.compile(r"\bffcs\b|\bslots?\b|\bregistration\b|\btime ?table\b|\badd.?drop\b|\bcredits?\b|"
                       r"\bcourses?\b|\bnptel\b|\bsummer sem|\bweekend sem|\belective", re.I),
    "rules": re.compile(r"\bid card\b|\bsecurity\b|\bred ?tag\b|\bgate\b|\bdress code\b|\bin.?time\b|"
                        r"\bcurfew\b|\bout.?pass\b|\bwarden\b|\bshorts\b|\brules?\b|\bfined?\b|\blate\b", re.I),
    "hostel": re.compile(r"\bhostels?\b|\bmess\b|\b[a-z] ?block\b|\broom|\blimra\b|\bfoodys\b|\bcanteen\b", re.I),
}

MODE_SEEDS = {           # which topics feed which mode, and how many examples we want
    "senior": (None, 1500),                                  # None = any question-like post
    "lawyer": (("attendance", "exams"), 600),
    "excuse": (("attendance", "exams", "hostel", "rules"), 600),
    "ffcs":   (("ffcs",), 600),
    "redtag": (("rules", "hostel"), 400),
}


def load_posts():
    with open(CLEAN_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def usable(p):
    text = f"{p['title']} {p['body']}"
    if p["flair"].strip().lower() in SKIP_FLAIRS:
        return False
    if SENSITIVE.search(text) or PEOPLE.search(text):
        return False
    # A harmless title can hide a tragedy that only the replies reveal ("Is VIT this messed up?" -> a death).
    if any(HARD.search(c["body"]) for c in p.get("top_comments", [])):
        return False
    if len(p["title"]) < 15 or "<link>" in p["title"]:
        return False
    return True


def is_soft(p):
    return bool(SOFT.search(f"{p['title']} {p['body']}"))


def question_like(p):
    t = p["title"].lower()
    return "?" in t or re.match(r"(is|are|can|should|how|what|which|why|when|where|does|do|will|any)\b", t) \
        or p["flair"].strip().lower() in {"ask vitians", "academics", "query", "freshers",
                                          "course & college queries", "hostel / mess / accommodation"}


REDTAG_FOCUS = re.compile(r"\b(security|guards?|id ?cards?|red ?tag|in.?time|curfew|late entry|outings?|"
                          r"(main|back|hostel) gate|gate ?pass|leave pass|warden|entry|sneak\w*|smuggl\w*|caught)\b", re.I)


def build_pools(posts):
    rng = random.Random(SEED)
    good = [p for p in posts if usable(p)]
    pools = {}
    for mode, (topics, want) in MODE_SEEDS.items():
        if topics is None:
            # /senior: questions, plus every SOFT post (people venting rarely phrase it as a question)
            cand = [p for p in good if question_like(p) or is_soft(p)]
        else:
            cand = [p for p in good if any(TOPIC[t].search(f"{p['title']} {p['body'][:500]}") for t in topics)]
        if USE_COMMENTS and mode in COMMENT_MODES:
            cand = [p for p in cand if senior_notes(p)]     # only posts where real seniors said something useful
        if USE_COMMENTS and mode == "redtag":
            # The broad rules/hostel pool gave redtag off-topic replies (laptop shops, mess bugs).
            # Keep posts about getting in/out and getting caught, so the replies are real gate/warden stories.
            cand = [p for p in cand if REDTAG_FOCUS.search(f"{p['title']} {p['body'][:500]}")]
        # Prefer posts people engaged with, but keep some randomness so it's not all memes.
        cand.sort(key=lambda p: p["score"] + p["num_comments"] + rng.random() * 10, reverse=True)
        top = cand[: want * 2]
        rng.shuffle(top)
        pools[mode] = top
    return good, pools


def seed_text(p, mode):
    body = re.sub(r"\s+", " ", p["body"]).strip()
    if mode == "senior":
        # Telegram users type short questions: mostly title only, sometimes a bit of context.
        h = int(hashlib.md5(p["id"].encode()).hexdigest(), 16)
        if body and h % 10 < 4:
            return f"{p['title']}\n{body[:300]}"
        return p["title"]
    return f"Title: {p['title']}\nPost: {body[:500] or '(no body)'}"


# --------------------------------------------------------------------------
# Teacher calls
# --------------------------------------------------------------------------
class FatalTeacherError(Exception):
    """An error every request will hit (bad key, bad option): stop instead of burning seeds."""


class BudgetReached(Exception):
    """--max-usd hit: stop cleanly, keep everything generated so far."""


class Teacher:
    def __init__(self, rps):
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        if not (self.base and self.key and self.model):
            sys.exit("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL first (see the top of this file).")
        self.gap = 1.0 / rps
        self.lock = threading.Lock()
        self.next_at = 0.0
        self.session = requests.Session()
        # Optional extra request fields, e.g. to keep a model's "thinking" short on OpenRouter:
        #   $env:LLM_EXTRA = '{"reasoning": {"effort": "low"}}'
        # ({"enabled": false} turns it off where allowed; GLM-5.3 refuses that.)
        self.extra = json.loads(os.environ.get("LLM_EXTRA", "{}"))
        self.tokens_in = self.tokens_out = self.calls = 0
        # JSON mode stops most broken-JSON replies. Set LLM_JSON_MODE=0 if a provider rejects it.
        self.json_mode = os.environ.get("LLM_JSON_MODE", "1") != "0"
        # Spend tracking for --max-usd. OpenRouter reports each request's exact cost when asked
        # ("usage": {"include": true}); for other providers set LLM_PRICE_IN / LLM_PRICE_OUT ($ per 1M tokens).
        self.cost = 0.0
        self.max_usd = None
        self.openrouter = "openrouter.ai" in self.base
        self.price_in = float(os.environ.get("LLM_PRICE_IN", "0") or 0)
        self.price_out = float(os.environ.get("LLM_PRICE_OUT", "0") or 0)

    def _throttle(self):
        with self.lock:
            now = time.time()
            wait = self.next_at - now
            self.next_at = max(now, self.next_at) + self.gap
        if wait > 0:
            time.sleep(wait)

    def chat(self, system, user, retries=6):
        delay = 5
        for _ in range(retries):
            if self.max_usd is not None and self.cost >= self.max_usd:
                raise BudgetReached(f"${self.cost:.3f} spent, cap is ${self.max_usd:.2f}")
            self._throttle()
            try:
                r = self.session.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"},
                    # Generous cap: reasoning models spend tokens "thinking" before the JSON answer.
                    json={"model": self.model, "temperature": 0.9, "max_tokens": 4000,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                          **({"response_format": {"type": "json_object"}} if self.json_mode else {}),
                          **({"usage": {"include": True}} if self.openrouter else {}),
                          **self.extra},
                    timeout=120)
            except requests.RequestException as e:
                print(f"  network error: {e}")
                time.sleep(delay); delay = min(delay * 2, 120); continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after", delay))
                print(f"  {r.status_code}; waiting {wait:.0f}s")
                time.sleep(wait); delay = min(delay * 2, 120); continue
            if 400 <= r.status_code < 500:
                # Bad key, no credits, wrong model name, unsupported option... retrying won't help.
                raise FatalTeacherError(f"{r.status_code}: {r.text[:300]}")
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
            data = r.json()
            u = data.get("usage") or {}
            cost = u.get("cost")
            if cost is None and (self.price_in or self.price_out):
                cost = (u.get("prompt_tokens", 0) * self.price_in + u.get("completion_tokens", 0) * self.price_out) / 1e6
            if cost is None and self.max_usd is not None:
                raise FatalTeacherError("can't track spend for --max-usd: provider didn't report cost. "
                                        "Set LLM_PRICE_IN and LLM_PRICE_OUT ($ per 1M tokens).")
            with self.lock:
                self.cost += float(cost or 0)
                self.calls += 1
                self.tokens_in += u.get("prompt_tokens", 0)
                self.tokens_out += u.get("completion_tokens", 0)   # includes hidden "thinking" tokens
            return data["choices"][0]["message"]["content"] or ""
        raise RuntimeError("teacher kept failing; try a lower --rps")

    def report(self):
        if self.calls and self.cost:
            print(f"\nSpent this run: ${self.cost:.3f} over {self.calls} calls "
                  f"(${1000 * self.cost / self.calls:.2f} per 1,000 calls)")
        if self.calls:
            print(f"\nTeacher usage: {self.calls} calls, {self.tokens_in:,} in / {self.tokens_out:,} out tokens "
                  f"(avg {self.tokens_in // self.calls} in / {self.tokens_out // self.calls} out per call). "
                  f"Multiply by your model's $/M price to estimate the full run.")


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else None


BANNED_OUT = re.compile(r"<user>|<link>|<phone>|<email>|<regno>|\bu/\w|https?://|"
                        r"\b\d{2}[A-Za-z]{3}\d{4,5}\b|"            # VIT reg no, e.g. 21BCE1234
                        r"(?<!\d)[6-9]\d{9}(?!\d)|"                  # phone
                        # "Prof. Sharma" / "Professor Rao" = a name; "prof section" / "professor. Then" = not
                        r"\b(?i:dr|prof|mr|mrs|ms)\.?\s+[A-Z][a-z]+|\b(?i:professor|sir|ma'?am)\s+[A-Z][a-z]+")
# Outputs get a slightly looser check than seeds: "VTOP dies" or a hospital medical-leave excuse is fine.
OUT_SENSITIVE = HARD   # generated replies must never contain HARD topics either


def to_turns(mode, seed, out):
    """Validate one teacher output -> (list of (user, assistant) turns or None, reason)."""
    if not isinstance(out, dict):
        return None, "bad json"
    if mode == "redtag":
        turns = [(t.get("user", ""), t.get("assistant", "")) for t in out.get("turns", []) if isinstance(t, dict)]
        if not 2 <= len(turns) <= 6:
            return None, f"{len(turns)} turns"
    elif mode == "senior":
        turns = [(seed, out.get("assistant", ""))]
    else:
        turns = [(out.get("user", ""), out.get("assistant", ""))]
    clean = []
    for u, a in turns:
        u, a = str(u).strip(), str(a).strip()
        if not u or not a:
            return None, "empty"
        if len(a.split()) > MAX_WORDS_MODE.get(mode, MAX_WORDS):
            return None, f"too long ({len(a.split())} words)"
        m = BANNED_OUT.search(a) or OUT_SENSITIVE.search(a) or (mode != "senior" and BANNED_OUT.search(u))
        if m:
            return None, f"banned: {m.group(0)!r}"
        clean.append((u, a))
    return clean, "ok"


def generate(mode, pool, teacher, want, workers):
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, f"{mode}.jsonl")
    done = set()
    kept = 0
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            done.add(r["seed_id"])
            kept += bool(r.get("turns"))
    todo = [p for p in pool if p["id"] not in done]
    print(f"[{mode}] have {kept} good / {len(done)} tried, want {want}; {len(todo)} seeds left")
    if kept >= want:
        return

    system = f"You write training data for a funny Telegram bot for VIT students.\n\n{LORE}\n\n{RULES}"
    write_lock = threading.Lock()

    def work(p):
        seed = seed_text(p, mode)
        rng = random.Random(f"{SEED}-{mode}-{p['id']}")   # reproducible spice per seed
        prompt = f"{TEACHER_TASK[mode]}\n\n--- spice ---\n{spice(mode, rng)}\n\n--- seed ---\n{seed}"
        notes = senior_notes(p) if (USE_COMMENTS and mode in COMMENT_MODES) else []
        if notes and mode in FACT_MODES:          # the notes block sets its own, longer length
            prompt = re.sub(r"\nLength: [^\n]*", "", prompt, count=1)
        if notes:
            header = NOTES_HEADER if mode in FACT_MODES else CREATIVE_NOTES_HEADER
            prompt += header + "\n".join(f"- {n}" for n in notes)
        try:
            raw = teacher.chat(system, prompt)
        except (FatalTeacherError, BudgetReached):
            raise
        except Exception as e:  # noqa: BLE001 - one bad seed shouldn't stop the run
            print(f"  [{mode}] {p['id']}: {e}")
            return None
        try:
            out = parse_json(raw)
        except (json.JSONDecodeError, TypeError):
            out = None
        turns, why = to_turns(mode, seed, out)
        rec = {"seed_id": p["id"], "mode": mode, "turns": turns, "notes_used": len(notes)}
        if turns is None:                       # keep the evidence so we can see why it failed
            rec.update(why=why, raw=raw[:2000])
        return rec

    f = open(path, "a", encoding="utf-8")
    with cf.ThreadPoolExecutor(workers) as ex:
        it = iter(todo)
        running = {ex.submit(work, p) for p in [next(it, None) for _ in range(workers)] if p}
        while running and kept < want:
            fin, running = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in fin:
                rec = fut.result()
                if rec:
                    with write_lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    kept += bool(rec["turns"])
                    if kept % 25 == 0 and rec["turns"]:
                        print(f"[{mode}] {kept}/{want}")
                nxt = next(it, None)
                if nxt and kept < want:
                    running.add(ex.submit(work, nxt))
    f.close()
    print(f"[{mode}] done: {kept} good examples")


# --------------------------------------------------------------------------
# Build train/val
# --------------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def build():
    rows, seen = [], set()
    # Re-check every seed against the CURRENT filters and clean data, so tightening a filter or
    # re-running clean.py (e.g. once comments are attached) drops examples without regenerating.
    posts = {p["id"]: p for p in load_posts()}
    dropped = recovered = 0
    for mode in BOT_SYSTEM:
        path = os.path.join(GEN_DIR, f"{mode}.jsonl")
        if not os.path.exists(path):
            continue
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if not r.get("turns"):
                # Discarded answers were already paid for: re-check them against the CURRENT limits and
                # filters (which have been loosened), instead of losing them.
                p0 = posts.get(r["seed_id"])
                if r.get("raw") and p0 is not None:
                    try:
                        r["turns"], _ = to_turns(mode, seed_text(p0, mode), parse_json(r["raw"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                    recovered += bool(r["turns"])
                if not r.get("turns"):
                    continue
            p = posts.get(r["seed_id"])
            if p is None or not usable(p):
                dropped += 1
                continue
            # Key on every user turn, not just the first: all redtag games open with "/redtag" or
            # "start", so a first-message key kept one game per opener and threw away the rest.
            key = (mode, norm(" | ".join(u for u, _ in r["turns"])))
            if key in seen:                       # same conversation twice -> keep one
                continue
            seen.add(key)
            msgs = [{"role": "system", "content": BOT_SYSTEM[mode]}]
            for u, a in r["turns"]:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            rows.append((r["seed_id"], mode, msgs))
    if dropped:
        print(f"  dropped {dropped} examples whose seed post no longer passes the safety filters")
    if recovered:
        print(f"  recovered {recovered} discarded answers that pass the current (looser) checks")
    if not rows:
        sys.exit("Nothing generated yet.")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_val = {}
    with open(os.path.join(TRAIN_DIR, "train.jsonl"), "w", encoding="utf-8") as tr, \
         open(os.path.join(TRAIN_DIR, "val.jsonl"), "w", encoding="utf-8") as va:
        for sid, mode, msgs in rows:
            is_val = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 1000 < VAL_FRACTION * 1000
            (va if is_val else tr).write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n_val.setdefault(mode, [0, 0])[is_val] += 1
    for mode, (t, v) in n_val.items():
        print(f"  {mode:7s} train {t:5d}  val {v:4d}")
    print(f"Wrote {TRAIN_DIR}/train.jsonl and val.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no API calls; show pools and sample prompts")
    ap.add_argument("--pilot", type=int, help="generate only N good examples per mode")
    ap.add_argument("--modes", default=",".join(BOT_SYSTEM), help="comma list, e.g. senior,lawyer")
    ap.add_argument("--rps", type=float, default=0.5, help="max teacher requests/sec (free tiers: keep low)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--build", action="store_true", help="only rebuild train/val")
    ap.add_argument("--gen-dir", default=GEN_DIR, help="where raw outputs go (use one per teacher to compare)")
    ap.add_argument("--use-comments", action="store_true",
                    help="give the teacher each post's top real replies (facts for /senior,/ffcs; creative "
                         "material for the rest). Run clean.py after the comments download first.")
    ap.add_argument("--max-usd", type=float, help="hard spend cap for THIS run; stops cleanly when reached")
    args = ap.parse_args()
    globals()["GEN_DIR"] = args.gen_dir
    globals()["USE_COMMENTS"] = args.use_comments

    if args.build:
        return build()

    posts = load_posts()
    good, pools = build_pools(posts)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"{len(posts):,} clean posts, {len(good):,} usable as seeds after safety filters")

    if args.dry_run:
        rng = random.Random(1)
        for m in modes:
            print(f"\n=== {m}: {len(pools[m])} seeds (want {MODE_SEEDS[m][1]}) ===")
            for p in rng.sample(pools[m], min(4, len(pools[m]))):
                print("  -", seed_text(p, m).replace("\n", " | ")[:160])
        return

    teacher = Teacher(args.rps)
    teacher.max_usd = args.max_usd
    try:
        for m in modes:
            want = args.pilot or MODE_SEEDS[m][1]
            generate(m, pools[m], teacher, want, args.workers)
    except FatalTeacherError as e:
        teacher.report()
        sys.exit(f"\nStopped: the API rejected the request, and retrying won't fix it.\n  {e}\n"
                 "Check LLM_API_KEY / LLM_MODEL / LLM_EXTRA, then run the same command again.")
    except BudgetReached as e:
        print(f"\nBudget cap reached ({e}). Everything generated so far is saved; "
              "re-run with a higher --max-usd to continue.")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
    teacher.report()
    if args.gen_dir == os.path.join("data", "gen"):
        build()


if __name__ == "__main__":
    main()
