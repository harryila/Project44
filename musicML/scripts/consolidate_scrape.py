"""Consolidate the scraped classical MusicXML (/tmp/scrape/*/musicxml) into a CLEAN solo-piano corpus:
  1. md5-dedup (same .krn converted by multiple agents -> identical output),
  2. keep SOLO PIANO/KEYBOARD only (<=2 score-parts; drops Lieder/quartets/multi-voice),
  3. LEAK-FILTER against the 14 ASAP TEST pieces (by filename AND embedded work/movement title) — the
     scraped craigsapp files are named systematically (e.g. beethoven sonata09/10 = Op.14 = ASAP test),
     so we exclude aggressively. Cheap xml.etree only (no music21); music21 parse happens at tokenize time.
Outputs a deduped, filtered list + copies to /tmp/scrape_clean/. Reports counts + leak hits."""
from __future__ import annotations
import glob, hashlib, os, re, shutil
import xml.etree.ElementTree as ET

SRC = "/tmp/scrape"
OUT = "/tmp/scrape_clean"

# ASAP TEST pieces -> exclusion regexes over "filename | work-title | movement-title" (lowercased).
# Aggressive (over-exclude is safe). craigsapp naming noted where relevant.
LEAK = [
    r"bwv.?_?846|wtc.*(prelude|fug).*(\b1\b|no.?1|c\b)|bach.*fug.*\bc\b.?maj",
    r"(beethoven).*(op.?14|sonata0?9\b|sonata10\b|sonata.*\bno.?(9|10)\b)",   # Op.14 = sonata09/10
    r"(chopin).*(ballad).*(\b1\b|no.?1|op.?23)|chopin.*023",
    r"reflets|(debussy).*(reflet|images.*(eau|water))",
    r"(haydn).*(sonata.*\b31\b|hob.*xvi.?31)",
    r"(liszt).*(gondol|venezia|napoli|annees.*(2|ii)|pelerinage.*(2|ii|ital))",
    r"\bk.?\s?332\b|(mozart).*(sonata12\b|sonata.*\bno.?12\b|sonata.*\b12\b)",   # K332 = sonata12
    r"(rachman).*(op.?23.*(\b4\b|no.?4)|prelude.*23.*4)",
    r"(schubert).*impromptu.*(op.?90|d.?899|\b1\b)",
    r"arabeske|(schumann).*arabesqu",
    r"(scriabin).*(op0?8\b|etude.*op.?8|op_?08)",          # Op.8 etudes (test = op8/11)
    r"(brahms).*(op.?118.*(\b2\b|no.?2)|intermezzo.*118)",
    r"(prokofiev).*toccata|toccata.*op.?11",
    r"ondine|gaspard",
]
RES = [re.compile(rx) for rx in LEAK]


def info(path):
    """(n_parts, title_text) via cheap xml parse. n_parts=-1 if unparseable."""
    try:
        # strip namespace noise by iterating
        nparts = 0
        title = []
        for ev, el in ET.iterparse(path, events=("start",)):
            tag = el.tag.split("}")[-1]
            if tag == "score-part":
                nparts += 1
            elif tag in ("work-title", "movement-title") and el.text:
                title.append(el.text)
            elif tag == "part" and nparts:   # parts started -> past the header, stop early
                break
        return nparts, " ".join(title).lower()
    except Exception:
        return -1, ""


def main():
    files = []
    for d in glob.glob(SRC + "/*/musicxml"):
        files += glob.glob(d + "/*.musicxml") + glob.glob(d + "/*.mxl")
    print(f"scanning {len(files)} scraped files")

    seen_md5 = set()
    kept, dup, nonpiano, leaked, bad = [], 0, 0, 0, 0
    os.makedirs(OUT, exist_ok=True)
    for f in files:
        try:
            h = hashlib.md5(open(f, "rb").read()).hexdigest()
        except Exception:
            bad += 1; continue
        if h in seen_md5:
            dup += 1; continue
        seen_md5.add(h)
        nparts, title = info(f)
        if nparts < 1:
            bad += 1; continue
        hay = (os.path.basename(f).lower() + " | " + title)
        if any(rx.search(hay) for rx in RES):
            leaked += 1; continue
        if nparts > 2:
            nonpiano += 1; continue          # solo-piano only (<=2 staves)
        kept.append(f)
    # copy kept with unique names
    for i, f in enumerate(kept):
        shutil.copy(f, os.path.join(OUT, f"scr_{i:06d}.musicxml"))
    print(f"kept(solo-piano, deduped, leak-clean)={len(kept)}  dup={dup}  nonpiano(>2 parts)={nonpiano}  "
          f"leaked={leaked}  bad={bad}")
    print(f"-> {OUT} ({len(kept)} files)")


if __name__ == "__main__":
    main()
