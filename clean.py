"""
Clean the raw r/Vit dump into a training-ready dataset.

Usage:
    python clean.py

Input:  data/raw/posts.jsonl, data/raw/comments.jsonl (comments optional)
Output: data/clean/posts.jsonl   one post per line, with its top comments attached
        data/clean/stats.txt     what got filtered and why

What it does:
  * drops deleted/removed posts, bots, NSFW, and near-empty posts
  * scrubs personal info: usernames, emails, phone numbers, VIT reg numbers, links
  * attaches the top-scoring top-level comments to each post (as context for later)
"""

import json
import os
import re
from collections import Counter, defaultdict

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "clean")
TOP_COMMENTS_PER_POST = 5

BOT_AUTHORS = {"automoderator", "[deleted]", "vit-bot", "remindmebot", "savevideo", "sneakpeekbot"}
DEAD_TEXT = {"[deleted]", "[removed]", "[ Removed by Reddit ]", "[removed by reddit]"}

# --- PII scrubbing ---------------------------------------------------------
PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "<email>"),
    (re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]{3,20}"), "<user>"),
    (re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "<phone>"),
    (re.compile(r"\b\d{2}[A-Za-z]{3}\d{4}\b"), "<regno>"),          # e.g. 21BCE1234
    (re.compile(r"https?://\S+"), "<link>"),
]


def scrub(text):
    if not text:
        return ""
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def is_bot(author):
    a = (author or "").lower()
    return a in BOT_AUTHORS or a.endswith("bot")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reasons = Counter()

    # 1. Posts
    posts = {}
    repost_of = {}  # (author, title) -> post id
    for p in read_jsonl(os.path.join(RAW_DIR, "posts.jsonl")):
        reasons["posts_total"] += 1
        title = (p.get("title") or "").strip()
        body = (p.get("selftext") or "").strip()
        if p["id"] in posts:
            reasons["dropped_duplicate"] += 1
            continue
        if p.get("removed_by_category") or title in DEAD_TEXT:
            reasons["dropped_removed"] += 1
            continue
        if is_bot(p.get("author")):
            reasons["dropped_bot"] += 1
            continue
        if p.get("over_18"):
            reasons["dropped_nsfw"] += 1
            continue
        if len(title) < 8:
            reasons["dropped_too_short"] += 1
            continue
        if body in DEAD_TEXT:
            body = ""
        # Same author posting the same title twice (reposts) -> keep the higher score.
        key = ((p.get("author") or "").lower(), title.lower())
        prev = repost_of.get(key)
        if prev is not None:
            reasons["dropped_repost"] += 1
            if (p.get("score") or 0) <= posts[prev]["score"]:
                continue
            del posts[prev]
        repost_of[key] = p["id"]
        posts[p["id"]] = {
            "id": p["id"],
            "created_utc": int(float(p["created_utc"])),
            "title": scrub(title),
            "body": scrub(body)[:4000],
            "score": p.get("score") or 0,
            "num_comments": p.get("num_comments") or 0,
            "flair": p.get("link_flair_text") or "",
            "top_comments": [],
        }
    reasons["posts_kept"] = len(posts)

    # 2. Comments -> keep the best top-level replies per post
    by_post = defaultdict(list)
    for c in read_jsonl(os.path.join(RAW_DIR, "comments.jsonl")):
        reasons["comments_total"] += 1
        post_id = (c.get("link_id") or "").removeprefix("t3_")
        if post_id not in posts:
            continue
        if c.get("parent_id") != c.get("link_id"):
            continue  # only top-level replies
        body = (c.get("body") or "").strip()
        if body in DEAD_TEXT or is_bot(c.get("author")) or len(body) < 3:
            continue
        by_post[post_id].append((c.get("score") or 0, scrub(body)[:1500]))

    for pid, items in by_post.items():
        items.sort(key=lambda x: x[0], reverse=True)
        posts[pid]["top_comments"] = [{"score": s, "body": b} for s, b in items[:TOP_COMMENTS_PER_POST]]
        reasons["comments_attached"] += len(posts[pid]["top_comments"])

    # 3. Write
    out_path = os.path.join(OUT_DIR, "posts.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in sorted(posts.values(), key=lambda x: x["created_utc"]):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = "\n".join(f"{k:22s} {v:,}" for k, v in reasons.items())
    with open(os.path.join(OUT_DIR, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(stats + "\n")
    print(stats)
    print(f"\nWrote {len(posts):,} clean posts -> {out_path}")


if __name__ == "__main__":
    main()
