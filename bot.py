"""
bot.py: VIT Bot on Telegram. Talks to any OpenAI-compatible server (llama-server, LM Studio).

    pip install -r requirements-bot.txt
    TELEGRAM_TOKEN=... python bot.py

Env vars:
  TELEGRAM_TOKEN   from @BotFather (required)
  LLM_URL          default http://127.0.0.1:8080/v1   (LM Studio: http://127.0.0.1:1234/v1)
  LLM_MODEL        default vit-bot (llama-server ignores it; LM Studio needs the model id)
  LLM_API_KEY      sent as a Bearer token when set (llama-server --api-key, e.g. the Modal deploy)
  MAX_PARALLEL     simultaneous model calls, default 1 (CPU servers: keep it low, others queue)

How it works: long polling (the bot asks Telegram for new messages), so the server needs no
open ports or domain. Each mode is one of the exact system prompts the model was trained on.
Messages about suicide/self-harm never reach the model; they get a fixed helpline reply.
"""
import asyncio
import logging
import os
import re

import httpx
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "vit-bot")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "1"))
REDTAG_MAX_TURNS = 8

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # don't log every Telegram poll
log = logging.getLogger("vitbot")

# Must match the training data word for word: the model only learned these prompts.
MODES = {
    "senior": "You are VIT Bot in /senior mode: a tired, sarcastic VIT Vellore senior answering a fresher. Roast lightly, then end with one genuinely useful tip. Under 100 words.",
    "lawyer": "You are VIT Bot in /lawyer mode: a dramatic attendance lawyer arguing the student's case to their faculty advisor, citing absurd 'precedents' and VIT rules. Under 100 words.",
    "excuse": "You are VIT Bot in /excuse mode: give one VIT-accurate excuse the student can actually say, delivered with a straight face. Under 80 words.",
    "ffcs": "You are VIT Bot in /ffcs mode: the FFCS Oracle. Read the student's timetable, slots or course plan like a horoscope: dramatic prophecy, one real tip. Under 100 words.",
    "redtag": "You are VIT Bot in /redtag mode: a text game. You are a Red Tag security anna at the VIT gate and the player has no ID card. Stay in character, react to their moves, keep replies short, and end the game with a clear win or loss after a few turns.",
}

HELP = (
    "VIT Bot 🦥 pick a mode:\n\n"
    "/senior <question> tired senior answers your fresher doubt\n"
    "/lawyer <your case> attendance lawyer fights your FA\n"
    "/excuse <situation> one straight-faced VIT excuse\n"
    "/ffcs <slots / timetable> FFCS horoscope\n"
    "/redtag text game: get past Red Tag security without your ID\n"
    "/stop end the current game\n\n"
    "Example: /lawyer 73% in DSA, missed 4 classes sick, no medical cert"
)

# Hard-coded guard: these never reach the model.
CRISIS = re.compile(
    r"\b(suicid\w*|kill(ing)? my ?self|end(ing)? (my|it) (life|all)|want(ed)? to die|wanna die|"
    r"self[- ]?harm\w*|cut(ting)? my ?self|hang(ing)? my ?self|no reason to live|"
    r"khudkushi|aatmahatya|atmahatya|marna chahta|marna chahti|mar jaana chahta|mar jana chahta)\b",
    re.I,
)
CRISIS_REPLY = (
    "Hey, I'm just a joke bot, but this matters more than any CAT or attendance number. 💛\n\n"
    "Please talk to someone right now:\n"
    "• Tele-MANAS (free, 24x7, many languages): 14416 or 1-800-891-4416\n"
    "• Emergency: 112\n"
    "• VIT student counselling / your hostel warden or FA\n\n"
    "You don't have to handle this alone."
)

games = {}                    # (chat_id, user_id) -> list of redtag messages
pending_mode = {}             # (chat_id, user_id) -> mode chosen with an empty command
llm_slots = asyncio.Semaphore(MAX_PARALLEL)
# 180 s covers a cold start on scale-to-zero hosts like Modal.
http = httpx.AsyncClient(timeout=180, headers={"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else None)


async def ask_llm(messages):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 250,
        "chat_template_kwargs": {"enable_thinking": False},  # trained to answer directly
    }
    async with llm_slots:
        r = await http.post(f"{LLM_URL}/chat/completions", json=payload)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


async def keep_typing(chat, stop):
    while not stop.is_set():
        await chat.send_action(ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def respond(update: Update, messages):
    stop = asyncio.Event()
    typing = asyncio.create_task(keep_typing(update.effective_chat, stop))
    try:
        reply = await ask_llm(messages)
    except Exception as e:
        log.warning("LLM error: %s", e)
        reply = "VTOP is down. Just kidding, my brain is. Try again in a minute."
    finally:
        stop.set()
        await typing
    await update.effective_message.reply_text(reply or "…", do_quote=True)
    return reply


async def guard(update: Update, text):
    """True if the message was a crisis message, answered here, and must not go to the model."""
    if CRISIS.search(text):
        log.info("crisis guard triggered")
        await update.effective_message.reply_text(CRISIS_REPLY)
        return True
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP)


async def stop_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    key = (update.effective_chat.id, update.effective_user.id)
    games.pop(key, None)
    pending_mode.pop(key, None)
    await update.effective_message.reply_text("Game over. Anna went back to his chai.")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = update.effective_message.text.split()[0][1:].split("@")[0].lower()
    text = " ".join(context.args).strip()
    key = (update.effective_chat.id, update.effective_user.id)

    if mode == "redtag":
        if text and await guard(update, text):
            return
        games[key] = [{"role": "system", "content": MODES["redtag"]}, {"role": "user", "content": text or "start"}]
        reply = await respond(update, games[key])
        games[key].append({"role": "assistant", "content": reply})
        return

    if not text:
        pending_mode[key] = mode
        await update.effective_message.reply_text(f"/{mode} mode. Send your message (reply to this in groups).")
        return
    if await guard(update, text):
        return
    await respond(update, [{"role": "system", "content": MODES[mode]}, {"role": "user", "content": text}])


async def plain_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    key = (update.effective_chat.id, update.effective_user.id)
    # In groups only react to replies to the bot (Telegram privacy mode hides the rest anyway).
    if update.effective_chat.type != ChatType.PRIVATE:
        if not (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id):
            return
    text = msg.text.strip()

    if key in games:
        if await guard(update, text):
            return
        games[key].append({"role": "user", "content": text})
        reply = await respond(update, games[key])
        games[key].append({"role": "assistant", "content": reply})
        if sum(m["role"] == "user" for m in games[key]) >= REDTAG_MAX_TURNS:
            games.pop(key)
            await msg.reply_text("(Game ended. /redtag to play again.)")
        return

    mode = pending_mode.pop(key, None)
    if mode is None:
        if update.effective_chat.type == ChatType.PRIVATE:
            await msg.reply_text(HELP)
        return
    if await guard(update, text):
        return
    await respond(update, [{"role": "system", "content": MODES[mode]}, {"role": "user", "content": text}])


def main():
    app = Application.builder().token(TOKEN).concurrent_updates(True).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("stop", stop_game))
    app.add_handler(CommandHandler(list(MODES), mode_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_message))
    log.info("VIT Bot polling, LLM at %s", LLM_URL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
