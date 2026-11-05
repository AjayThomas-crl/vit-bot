"""
Find out what r/Vit actually talks about -> decides which bot modes to build.

Usage:
    python explore.py

Input:  data/clean/posts.jsonl
Output: data/reports/topics.md   (open it in VS Code / any Markdown viewer)

Two views:
  1. Hand-picked VIT topics (FFCS, hostel, mess, attendance, ...) with counts,
     engagement, and the top posts in each -> maps directly to bot modes.
  2. Auto-discovered clusters (needs scikit-learn) -> topics we didn't think of.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict

IN_PATH = os.path.join("data", "clean", "posts.jsonl")
OUT_DIR = os.path.join("data", "reports")

TOPICS = {
    "FFCS / registration": r"\bffcs\b|\bslots?\b|\bregistration\b|\btime ?table\b|\badd.?drop\b|\bcredits?\b",
    "Faculty / profs": r"\bfacult|\bprof\b|\bprofessors?\b|\bteachers?\b|\bsir\b|\bma'?am\b|\bproctor",
    "Attendance / debar": r"\battendance\b|\bdebar|\b75 ?%|\bproxy\b|\bbunk|\bod\b|\bon.?duty\b|\bmedical leave\b",
    "Exams (CAT/FAT/grades)": r"\bcat[- ]?(?:1|2|i{1,2})\b|\bfat\b|\bexams?\b|\bgrad(?:e|es|ing)\b|\bc?gpa\b|\bre.?fat\b|\barrears?\b|\bbacklogs?\b",
    "Hostel / curfew": r"\bhostels?\b|\b[a-z] ?block\b|\bwarden|\bcurfew\b|\broom ?mates?\b|\bmh\b|\blh\b|\bin.?time\b|\bout.?pass\b",
    "Mess / food": r"\bmess\b|\bfood\b|\bcanteen\b|\blimra\b|\bfoodys\b|\bgazebo\b|\bbiryani\b|\bmaggi\b",
    "Placements / internships": r"\bplacements?\b|\bplaced\b|\bintern|\bpackage\b|\blpa\b|\bsuper dream\b|\brecruit|\bctc\b",
    "Admissions / VITEEE": r"\bviteee\b|\badmissions?\b|\bcounsell?ing\b|\bcategory [1-5]\b|\bfees?\b|\bbranch change\b|\branks?\b",
    "Fests / clubs": r"\bgravitas\b|\briviera\b|\bclubs?\b|\bchapters?\b|\bfests?\b|\bhackathon|\byantra\b",
    "VTOP / tech issues": r"\bvtop\b|\bportal\b|\bwifi\b|\bwi-fi\b|\bvolsbb\b|\binternet\b",
    "Campus / city life": r"\bkatpadi\b|\bbus\b|\bauto\b|\btrip\b|\bweekend\b|\bheat\b|\bweather\b|\bmall\b|\bvellore\b",
    "Rules / admin rants": r"\bdress code\b|\bid card\b|\bsecurity\b|\brules?\b|\bfined?\b|\badmin\b|\bmanagement\b|\bsuspend|\bred ?tag\b",
    "Mental health / stress": r"\bstress|\bdepress|\blonely\b|\banxiety\b|\bburn.?out\b|\bhomesick",
    "Campus wars (Chennai/AP/Bhopal)": r"\bchennai\b|\bbhopal\b|\bvit.?ap\b|\bamaravati\b",
}
TOPIC_RE = {k: re.compile(v, re.I) for k, v in TOPICS.items()}

STOP = set("""a an the and or but if to of in on at for with from by is are was were be been am i me my
we our you your he she it they them this that these those what which who how why when where can
could would should do does did doing have has had will just so not no yes any anyone someone
there here about into out up down over than then also get got vit vellore please help guys
anybody know need want like one two its im don dont""".split())


def load():
    with open(IN_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def year(p):
    return time.gmtime(p["created_utc"]).tm_year


def topic_view(posts):
    lines = ["## 1. VIT topics (hand-picked)\n",
             "| Topic | Posts | % of all | Avg score | Avg comments |",
             "|---|---:|---:|---:|---:|"]
    buckets = defaultdict(list)
    for p in posts:
        text = f"{p['title']} {p['body'][:500]}"
        for name, rx in TOPIC_RE.items():
            if rx.search(text):
                buckets[name].append(p)
    ranked = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for name, ps in ranked:
        n = len(ps)
        lines.append(f"| {name} | {n:,} | {100*n/len(posts):.1f}% | "
                     f"{sum(p['score'] for p in ps)/n:.1f} | {sum(p['num_comments'] for p in ps)/n:.1f} |")

    lines.append("\n### Top posts per topic (by score)\n")
    for name, ps in ranked:
        lines.append(f"**{name}**\n")
        for p in sorted(ps, key=lambda x: x["score"], reverse=True)[:8]:
            lines.append(f"- ({p['score']}) {p['title']}")
        lines.append("")
    return lines


def overview(posts):
    per_year = Counter(year(p) for p in posts)
    flairs = Counter(p["flair"] for p in posts if p["flair"])
    words = Counter()
    for p in posts:
        for w in re.findall(r"[a-z][a-z']+", p["title"].lower()):
            if w not in STOP and len(w) > 2:
                words[w] += 1
    lines = ["## 0. Overview\n",
             f"- Clean posts: **{len(posts):,}**",
             f"- Posts with comments attached: **{sum(1 for p in posts if p['top_comments']):,}**",
             "", "**Posts per year:** " + ", ".join(f"{y}: {n:,}" for y, n in sorted(per_year.items())),
             "", "**Top flairs:** " + ", ".join(f"{f} ({n})" for f, n in flairs.most_common(15)),
             "", "**Most common title words:** " + ", ".join(f"{w} ({n})" for w, n in words.most_common(60)),
             "", "### All-time top 25 posts\n"]
    for p in sorted(posts, key=lambda x: x["score"], reverse=True)[:25]:
        lines.append(f"- ({p['score']}, {year(p)}) {p['title']}")
    lines.append("")
    return lines


def cluster_view(posts, k=20):
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return ["## 2. Auto-discovered clusters\n",
                "_Skipped: run `pip install scikit-learn` to enable._\n"]
    docs = [f"{p['title']} {p['body'][:300]}" for p in posts]
    vec = TfidfVectorizer(stop_words=list(STOP | {"vit", "vellore"}), max_features=20000,
                          min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(docs)
    k = min(k, max(2, len(posts) // 50))
    # k-means straight on 20k-dim sparse TF-IDF collapses into one giant cluster.
    # Project to 100 dense "topic" dims first (LSA), re-normalize, then cluster.
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    svd = TruncatedSVD(n_components=100, random_state=42)
    Z = normalize(svd.fit_transform(X))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=4096).fit(Z)
    centers = svd.inverse_transform(km.cluster_centers_)  # back to word space for labels
    terms = vec.get_feature_names_out()
    lines = ["## 2. Auto-discovered clusters (TF-IDF + LSA + k-means)\n"]
    sizes = Counter(km.labels_)
    for c, size in sizes.most_common():
        top_terms = [terms[i] for i in centers[c].argsort()[::-1][:10]]
        members = [posts[i] for i, l in enumerate(km.labels_) if l == c]
        lines.append(f"**Cluster {c}** — {size:,} posts — _{', '.join(top_terms)}_\n")
        for p in sorted(members, key=lambda x: x["score"], reverse=True)[:5]:
            lines.append(f"- ({p['score']}) {p['title']}")
        lines.append("")
    return lines


def main():
    posts = load()
    if not posts:
        print("No posts found. Run scrape_vit.py and clean.py first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    md = ["# What r/Vit talks about\n"] + overview(posts) + topic_view(posts) + cluster_view(posts)
    out = os.path.join(OUT_DIR, "topics.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report written -> {out}")


if __name__ == "__main__":
    main()
