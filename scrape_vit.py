"""
Download the full history of r/Vit (posts + comments) from the Arctic Shift archive.

Usage:
    python scrape_vit.py                 # posts + comments, full history
    python scrape_vit.py --only posts    # just posts (much faster)
    python scrape_vit.py --after 2022-01-01

Output (JSONL, one Reddit object per line):
    data/raw/posts.jsonl
    data/raw/comments.jsonl

The script is resumable: stop it any time (Ctrl+C) and run it again —
it continues from the newest item already saved.
"""

import argparse
import json
import os
import sys
import time

import requests

BASE_URL = "https://arctic-shift.photon-reddit.com"
SUBREDDIT = "Vit"
RAW_DIR = os.path.join("data", "raw")

# Only ask for the fields we actually use -> smaller, faster responses.
FIELDS = {
    "posts": [
        "id", "author", "created_utc", "title", "selftext", "score",
        "num_comments", "link_flair_text", "url", "over_18",
    ],
    "comments": [
        "id", "author", "created_utc", "body", "score", "link_id", "parent_id",
    ],
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "vitbot-dataset-builder/1.0 (personal learning project)"


def last_saved(path):
    """Return (set of ids, newest created_utc) already in the output file."""
    ids, newest = set(), None
    if not os.path.exists(path):
        return ids, newest
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line from a crash; ignore it
            ids.add(obj["id"])
            ts = int(float(obj["created_utc"]))
            newest = ts if newest is None else max(newest, ts)
    return ids, newest


def fetch_page(kind, after, limit, max_retries=8, use_fields=True):
    """One API call. Handles rate limits (429) and transient errors with backoff."""
    params = {
        "subreddit": SUBREDDIT,
        "after": after,
        "sort": "asc",
        "limit": limit,
    }
    if use_fields:
        params["fields"] = ",".join(FIELDS[kind])
    url = f"{BASE_URL}/api/{kind}/search"
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue

        if r.status_code == 429:
            wait = r.headers.get("X-RateLimit-Reset")
            wait = int(float(wait)) + 1 if wait else delay
            print(f"  rate limited; sleeping {wait}s")
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        if r.status_code >= 500:
            print(f"  server error {r.status_code}; retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if r.status_code == 422:
            # Seen once at a busy stretch (May 2024). Usually transient: back off,
            # and step the page size down auto -> 100 -> 50 -> 25.
            print(f"  archive said 422 (limit={limit}): {r.text[:300]}")
            smaller = {"auto": 100, 100: 50, 50: 25}.get(limit)
            time.sleep(delay)
            delay = min(delay * 2, 120)
            if smaller:
                limit = smaller
                params["limit"] = limit
            continue
        if r.status_code == 400:
            print(f"  archive said 400: {r.text[:300]}")
            if limit == "auto":
                # Some servers reject "auto" under load; fall back to fixed pages.
                return fetch_page(kind, after, 100, max_retries, use_fields)
            if use_fields:
                # A field name may have changed; ask for everything instead.
                print("  retrying without the field filter")
                return fetch_page(kind, after, limit, max_retries, use_fields=False)
        r.raise_for_status()

        payload = r.json()
        if isinstance(payload, dict):
            if payload.get("error"):
                raise RuntimeError(f"API error: {payload['error']}")
            return payload.get("data") or []
        return payload  # in case the API ever returns a bare list
    raise RuntimeError(f"Giving up after {max_retries} attempts ({url})")


def download(kind, start_after):
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"{kind}.jsonl")
    seen, newest = last_saved(out_path)
    after = newest if newest is not None else start_after
    if seen:
        print(f"[{kind}] resuming: {len(seen):,} already saved, continuing after {after}")
    else:
        print(f"[{kind}] starting fresh from {after}")

    total_new = 0
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        while True:
            page = fetch_page(kind, after, "auto")
            new_items = [it for it in page if it.get("id") not in seen]
            if not new_items:
                break  # reached the end of the archive

            for it in new_items:
                seen.add(it["id"])
                out.write(json.dumps(it, ensure_ascii=False) + "\n")
            out.flush()
            total_new += len(new_items)

            after = max(int(float(it["created_utc"])) for it in page)
            day = time.strftime("%Y-%m-%d", time.gmtime(after))
            rate = total_new / max(time.time() - started, 1)
            print(f"[{kind}] +{len(new_items):4d}  total {len(seen):,}  up to {day}  ({rate:.0f}/s)")
            time.sleep(0.5)  # be polite: ~2 requests/second max

    print(f"[{kind}] done. {total_new:,} new, {len(seen):,} total -> {out_path}\n")


def main():
    global BASE_URL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["posts", "comments"], help="download just one kind")
    ap.add_argument("--after", default="2008-01-01", help="start date (default: beginning)")
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)  # for testing
    args = ap.parse_args()

    BASE_URL = args.base_url.rstrip("/")

    kinds = [args.only] if args.only else ["posts", "comments"]
    try:
        for kind in kinds:
            download(kind, args.after)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
