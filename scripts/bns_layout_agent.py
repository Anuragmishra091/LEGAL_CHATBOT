#!/usr/bin/env python3
"""
bns_layout_agent.py
-------------------
Chunking script for Bharatiya Nyaya Sanhita (BNS) 2023.

Strategy: Section-level chunking with Illustration Binding.
  - Atomic unit  = one complete BNS section (body + sub-sections +
                   Explanations + Illustrations + Exceptions + Provisos).
  - No token-overlap.  Cross-reference metadata is used for semantic linking.
  - Oversized sections (> TOKEN_THRESHOLD tokens) are split ONLY at
    sub-section (n) boundaries; each split part carries the section heading
    as a prefix so it remains self-contained.

Output → data/bns_chunks_metadata.json
         data/bns_extracted_text.txt   (debug: raw extracted text)
"""

import re
import json
import fitz                      # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PDF_PATH        = BASE_DIR / "data" / "BNS.pdf"
OUTPUT_PATH     = BASE_DIR / "data" / "bns_chunks_metadata.json"
DEBUG_TXT_PATH  = BASE_DIR / "data" / "bns_extracted_text.txt"

TOKEN_THRESHOLD = 800      # approx tokens; sections above this are split
MAX_SECTION_NUM = 400      # BNS has 358 sections; cap false-positives above 400

# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class BNSChunk:
    chunk_id : str
    prefix   : str    # "Chapter X — Title | Section N | Section Title"
    text     : str    # section body (or one split-part of it)
    metadata : dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PDF → raw text
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(path: Path) -> str:
    """
    Extract text from a gazette-format PDF.

    Gazette PDFs typically have a two-column layout:
      • Left strip  (~0–22 % of page width): marginal notes (section titles)
      • Right strip (~22–100 %):             section body text

    We extract all text blocks, sort by vertical then horizontal position,
    and join them.  This keeps marginal notes immediately before their
    corresponding body text in the stream — which lets the section regex
    find both the number *and* the title on adjacent lines.
    """
    doc   = fitz.open(str(path))
    pages: List[str] = []

    for page in doc:
        pw     = page.rect.width
        blocks = page.get_text("blocks")          # (x0,y0,x1,y1,text,bno,btype)
        tb     = [b for b in blocks if b[6] == 0 and b[4].strip()]
        if not tb:
            continue

        # Sort: primary key = row (y rounded to 4 pt grid), secondary = left-edge (x)
        tb.sort(key=lambda b: (round(b[1] / 4) * 4, b[0]))
        page_text = "\n".join(b[4].strip() for b in tb)
        pages.append(page_text)

    doc.close()
    return "\n\n".join(pages)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Clean extracted text
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_PATTERNS = re.compile(
    r"(THE GAZETTE OF INDIA|MINISTRY OF LAW AND JUSTICE|"
    r"Hkkjr dk jkti=|भारत का राजपत्र|"
    r"SEC\.\s*1\]|PART\s+II[—–-]|"       # gazette section headers
    r"^\s*\d{1,4}\s*$)",                  # lone page numbers
    re.IGNORECASE | re.MULTILINE,
)

def clean_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        if not _SKIP_PATTERNS.search(line):
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Document structure parsers
# ─────────────────────────────────────────────────────────────────────────────

# Chapter heading — "CHAPTERI" (no space!) then title on next line
# Also covers "CHAPTER X" (with space) and "CHAPTER I — TITLE" (inline)
_RE_CHAPTER = re.compile(
    r"(?:^|\n)\s*CHAPTER\s*([IVXLCDM]+)\s*\n\s*([^\n]+)",
    re.MULTILINE,
)

# Inline chapter: "CHAPTER I — PRELIMINARY"
_RE_CHAPTER_INLINE = re.compile(
    r"(?:^|\n)\s*CHAPTER\s*([IVXLCDM]+)\s*[-–—]+\s*([^\n]+)",
    re.MULTILINE,
)

# Section heading — gazette standard: "303. Theft.—" (inline marginal note)
_RE_SECTION_INLINE = re.compile(
    r"(?:^|\n)\s*(\d{1,3})\.\s+((?:[A-Z][^.—–\n]{0,120}))[.—–]",
    re.MULTILINE,
)

# Section heading — bare number at line start (BNS gazette: marginal note
# appears as a separate embedded block, not inline with the section number).
# Uses [ \t]* (zero or more spaces) to catch PDF extraction artefacts where
# the space after the period is lost, e.g. "104.Whoever" instead of "104. Whoever".
_RE_SECTION_BARE = re.compile(
    r"(?:^|\n)[ \t]*(\d{1,3})\.[ \t]*(?=[^\n\d])",
    re.MULTILINE,
)

# Content markers
_RE_SUBSEC    = re.compile(r"\(\d+\)\s+",         re.MULTILINE)
_RE_EXPL      = re.compile(r"\bExplanation\b",     re.IGNORECASE)
_RE_ILLUS     = re.compile(r"\bIllustrations?\b",  re.IGNORECASE)
_RE_EXCEPT    = re.compile(r"\bException\b\s*\d*", re.IGNORECASE)
_RE_PROVISO   = re.compile(r"\bProvided\s+that\b", re.IGNORECASE)
_RE_XREF      = re.compile(r"\bsections?\s+(\d{1,3})\b", re.IGNORECASE)
_RE_PUNISH    = re.compile(
    r"((?:rigorous|simple)\s+imprisonment\s+for[^,;.\n]{0,80}|"
    r"imprisonment\s+(?:for\s+life|for\s+a\s+term[^,;.\n]{0,60})|"
    r"\bdeath\b|"
    r"fine(?:\s+(?:which\s+may\s+extend|not\s+less\s+than)[^,;.\n]{0,60})?)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  Helper: Roman numeral → integer
# ─────────────────────────────────────────────────────────────────────────────

def roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    n, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        n += v if v >= prev else -v
        prev = v
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  Parse section headings from text
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawSection:
    chapter_roman : str
    chapter_title : str
    section_num   : int
    section_title : str   # from marginal note
    raw_text      : str   # body text (everything after "N. Title.—" up to next section)


def _collect_chapters(text: str) -> List[Tuple[int, str, str]]:
    """Return list of (char_pos, roman, title) for every chapter heading."""
    chapters: List[Tuple[int, str, str]] = []
    seen_positions: set = set()

    for m in _RE_CHAPTER.finditer(text):
        if m.start() not in seen_positions:
            # group(2) is the line immediately after CHAPTERXYZ
            title = m.group(2).strip()
            # Skip sub-headers that start with 'Of ' / 'of ' — those are
            # section-group headings, not chapter titles
            if not re.match(r'^[Oo]f ', title):
                chapters.append((m.start(), m.group(1).strip(), title))
                seen_positions.add(m.start())

    for m in _RE_CHAPTER_INLINE.finditer(text):
        if m.start() not in seen_positions:
            chapters.append((m.start(), m.group(1).strip(), m.group(2).strip()))
            seen_positions.add(m.start())

    chapters.sort(key=lambda x: x[0])
    return chapters


# ── Section title (marginal note) extractor ───────────────────────────────────

# Words that start body sentences — not marginal notes
_BODY_STARTERS = re.compile(
    r'^(In |The |Whoever |Nothing |For |A |An |Every |When |Where |No |'
    r'Any |This |That |With |Upon |After |Before |During |If |Of |\(|\d|'
    r'CHAPTER)',
    re.IGNORECASE,
)

def _extract_section_title(
    full_text: str, pos: int, body_start: int, default: str
) -> str:
    """
    Recover the marginal note (section title) for a section.

    BNS gazette: marginal notes appear as isolated short text blocks,
    either immediately BEFORE the section number (extracted first due to
    lower x-position in two-column layout) or EMBEDDED within the first
    ~400 chars of the body text.

    Strategy
    --------
    1. Look in the 250 chars that precede the section number.
    2. Scan the first 400 chars of the body for short isolated lines.
    3. Fallback: first 8 words of the body.
    """
    # ── Strategy 1: text before the section number ─────────────────────
    preceding = full_text[max(0, pos - 250):pos]
    prev_lines = [l.strip() for l in preceding.split('\n') if l.strip()]
    for line in reversed(prev_lines[-6:]):
        words = line.split()
        if (
            1 <= len(words) <= 10          # allow single-word titles like "Definitions."
            and line[0].isupper()
            and not line.isupper()         # skip ALL-CAPS chapter lines
            and not _BODY_STARTERS.match(line)
            and not re.match(r'^CHAPTER', line)
        ):
            return line.rstrip('.')

    # ── Strategy 2: embedded short lines in body ────────────────────────
    body_window = full_text[body_start: body_start + 400]
    lines = body_window.split('\n')

    # Skip the very first line (it IS the section body, not the title)
    candidate_parts: List[str] = []
    for line in lines[1:]:   # start from line 2 of the body
        line = line.strip()
        if not line:
            if candidate_parts:
                break
            continue
        words = line.split()
        if (
            1 <= len(words) <= 5
            and line[0].isupper()
            and not line.isupper()
            and not _BODY_STARTERS.match(line)
            and not re.match(r'^(Explanation|Illustration|Exception|Proviso)', line, re.I)
            and not line.startswith('(')
        ):
            candidate_parts.append(line.rstrip('.,'))
            if len(candidate_parts) >= 4:   # max 4 lines for a marginal note
                break
        else:
            if candidate_parts:
                break

    if candidate_parts:
        return ' '.join(candidate_parts)

    # ── Strategy 3: fallback ────────────────────────────────────────────
    return default


def _chapter_at(pos: int, chapters: List[Tuple[int, str, str]]) -> Tuple[str, str]:
    roman, title = "0", "Preamble"
    for cpos, r, t in chapters:
        if cpos <= pos:
            roman, title = r, t
        else:
            break
    return roman, title


def _monotone_filter(
    raw: List[Tuple[int, int, str, int]]
) -> List[Tuple[int, int, str, int]]:
    """
    Keep only entries whose section number is part of a forward-monotone
    sequence.  This removes false positives from inline cross-references
    (e.g. "section 5" within section 100's body text cannot appear at
    line-start as a bare number, so false positives are rare).
    """
    filtered: List[Tuple[int, int, str, int]] = []
    last = 0
    for item in raw:
        num = item[1]
        if 1 <= num <= MAX_SECTION_NUM and num >= last and num <= last + 100:
            filtered.append(item)
            last = num
    return filtered


def parse_document(text: str) -> List[RawSection]:
    """
    Two-pass parse:
      Pass 1 — find chapter headings and their positions.
      Pass 2 — find section starts with the bare pattern (broadest match);
               fall back to inline pattern if bare finds too few.
      Pass 3 — extract section titles from surrounding context.
      Then slice body text for each section.
    """
    chapters = _collect_chapters(text)

    # ── Bare section starts (primary — BNS gazette format) ───────────────────
    raw_bare: List[Tuple[int, int, str, int]] = []
    for m in _RE_SECTION_BARE.finditer(text):
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        raw_bare.append((m.start(), num, "", m.end()))   # title filled below

    bare_filtered = _monotone_filter(raw_bare)

    # ── Inline section starts (fallback for nicely formatted PDFs) ───────────
    raw_inline: List[Tuple[int, int, str, int]] = []
    for m in _RE_SECTION_INLINE.finditer(text):
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        title = m.group(2).strip().rstrip(".")
        raw_inline.append((m.start(), num, title, m.end()))

    inline_filtered = _monotone_filter(raw_inline)

    # ── Choose the richer parse ───────────────────────────────────────────────
    chosen = bare_filtered if len(bare_filtered) >= len(inline_filtered) else inline_filtered
    has_inline_titles = (chosen is inline_filtered)

    if not chosen:
        print("  WARNING: No section headings detected. "
              "Inspect data/bns_extracted_text.txt and tune the regex patterns.")
        return []

    # ── Slice body text and extract titles ────────────────────────────────────
    sections: List[RawSection] = []
    for i, (pos, num, raw_title, body_start) in enumerate(chosen):
        end_pos  = chosen[i + 1][0] if i + 1 < len(chosen) else len(text)
        raw_text = text[body_start:end_pos].strip()
        roman, ch_title = _chapter_at(pos, chapters)

        # Determine section title
        if has_inline_titles and raw_title:
            title = raw_title
        else:
            default = f"Section {num}"
            title = _extract_section_title(text, pos, body_start, default)

        sections.append(RawSection(
            chapter_roman = roman,
            chapter_title = ch_title,
            section_num   = num,
            section_title = title,
            raw_text      = raw_text,
        ))

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Token counting  (no extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """~1 token per 4 characters (OpenAI BPE heuristic). Always ≥ 1."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_punishment(text: str) -> str:
    hits = []
    seen: set = set()
    for m in _RE_PUNISH.finditer(text):
        val = " ".join(m.group(0).split())[:120]  # normalise whitespace
        if val.lower() not in seen:
            hits.append(val)
            seen.add(val.lower())
    return "; ".join(hits)


def _extract_cross_refs(text: str, own_num: int) -> List[int]:
    refs: set = set()
    for m in _RE_XREF.finditer(text):
        try:
            n = int(m.group(1))
            if n != own_num and 1 <= n <= MAX_SECTION_NUM:
                refs.add(n)
        except ValueError:
            pass
    return sorted(refs)


def make_metadata(sec: RawSection, text: str, split_part: int = 0) -> dict:
    return {
        "chapter_num"      : roman_to_int(sec.chapter_roman) if sec.chapter_roman not in ("0", "") else 0,
        "chapter_roman"    : sec.chapter_roman,
        "chapter_title"    : sec.chapter_title,
        "section_num"      : sec.section_num,
        "section_title"    : sec.section_title,
        "has_illustration" : bool(_RE_ILLUS.search(text)),
        "has_exception"    : bool(_RE_EXCEPT.search(text)),
        "has_explanation"  : bool(_RE_EXPL.search(text)),
        "has_proviso"      : bool(_RE_PROVISO.search(text)),
        "has_definition"   : (
            '"' in text or "\u201c" in text
            or bool(re.search(r'\bmeans\b', text, re.IGNORECASE))
        ),
        "punishment_range" : _extract_punishment(text),
        "cross_refs"       : _extract_cross_refs(text, sec.section_num),
        "token_count"      : approx_tokens(text),
        "split_part"       : split_part,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Split oversized sections at sub-section boundaries
# ─────────────────────────────────────────────────────────────────────────────

def split_at_subsections(sec: RawSection, threshold: int) -> List[dict]:
    """
    Greedily group sub-sections into parts, flushing each time the running
    token count would exceed *threshold*.

    Returns a list of partial-chunk dicts (prefix, text, meta).
    """
    text   = sec.raw_text
    prefix = (
        f"Chapter {sec.chapter_roman} — {sec.chapter_title} | "
        f"Section {sec.section_num} | {sec.section_title}"
    )

    # Positions where a new sub-section "(n) " starts
    boundaries = [m.start() for m in _RE_SUBSEC.finditer(text)]

    if not boundaries:
        # Nothing to split on — emit as single oversized chunk
        return [{"prefix": prefix, "text": text,
                 "meta": make_metadata(sec, text, split_part=0)}]

    # Build non-overlapping segments: [0, b0), [b0, b1), [b1, b2), ...
    seg_starts = [0] + boundaries
    seg_ends   = boundaries + [len(text)]

    parts: List[str] = []
    group_start = 0     # index into seg_starts for current group's first segment
    running_len = 0

    for i in range(len(seg_starts)):
        seg_len = seg_ends[i] - seg_starts[i]

        if running_len + seg_len > threshold and i > group_start:
            # Flush the current group (segments group_start … i-1)
            part_text = text[seg_starts[group_start]:seg_starts[i]].strip()
            if part_text:
                parts.append(part_text)
            group_start  = i
            running_len  = seg_len
        else:
            running_len += seg_len

    # Flush the last group
    tail = text[seg_starts[group_start]:].strip()
    if tail:
        parts.append(tail)

    # Assign split_part indices; if only one part came out, mark it as 0 (unsplit)
    result: List[dict] = []
    total = len(parts)
    for idx, part_text in enumerate(parts, start=1):
        result.append({
            "prefix": prefix,
            "text"  : part_text,
            "meta"  : make_metadata(sec, part_text,
                                    split_part=idx if total > 1 else 0),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Assemble final chunks
# ─────────────────────────────────────────────────────────────────────────────

def assemble_chunks(sections: List[RawSection], threshold: int) -> List[BNSChunk]:
    chunks: List[BNSChunk] = []

    for sec in sections:
        prefix = (
            f"Chapter {sec.chapter_roman} — {sec.chapter_title} | "
            f"Section {sec.section_num} | {sec.section_title}"
        )

        if approx_tokens(sec.raw_text) <= threshold:
            chunks.append(BNSChunk(
                chunk_id = f"bns_s{sec.section_num:03d}",
                prefix   = prefix,
                text     = sec.raw_text,
                metadata = make_metadata(sec, sec.raw_text),
            ))
        else:
            parts = split_at_subsections(sec, threshold)
            needs_suffix = len(parts) > 1
            for i, p in enumerate(parts, start=1):
                cid = (f"bns_s{sec.section_num:03d}_p{i}"
                       if needs_suffix
                       else f"bns_s{sec.section_num:03d}")
                chunks.append(BNSChunk(
                    chunk_id = cid,
                    prefix   = p["prefix"],
                    text     = p["text"],
                    metadata = p["meta"],
                ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Save output
# ─────────────────────────────────────────────────────────────────────────────

def save_chunks(chunks: List[BNSChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in chunks], fh, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks)} chunks → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PDF    : {PDF_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print()

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    print("[1/4] Extracting text from PDF …")
    raw_text = extract_pdf_text(PDF_PATH)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("[2/4] Cleaning text …")
    text = clean_text(raw_text)
    DEBUG_TXT_PATH.write_text(text, encoding="utf-8")
    print(f"  Debug text saved → {DEBUG_TXT_PATH}")

    # ── Step 3: Parse ─────────────────────────────────────────────────────────
    print("[3/4] Parsing document structure …")
    sections = parse_document(text)

    if not sections:
        print("\nERROR: 0 sections parsed.")
        print("Open data/bns_extracted_text.txt and check what the text looks like,")
        print("then adjust the _RE_SECTION_INLINE or _RE_SECTION_BARE patterns.")
        return

    print(f"  Sections found   : {len(sections)}")
    print(f"  Section range    : {sections[0].section_num} – {sections[-1].section_num}")
    chapters_seen = sorted(set(s.chapter_roman for s in sections),
                           key=lambda r: roman_to_int(r))
    print(f"  Chapters         : {', '.join(chapters_seen)}")

    # ── Step 4: Chunk ─────────────────────────────────────────────────────────
    print(f"[4/4] Assembling chunks (threshold = {TOKEN_THRESHOLD} tokens) …")
    chunks = assemble_chunks(sections, TOKEN_THRESHOLD)

    oversized = sum(1 for s in sections if approx_tokens(s.raw_text) > TOKEN_THRESHOLD)
    print(f"  Sections split   : {oversized}")
    print(f"  Total chunks     : {len(chunks)}")

    save_chunks(chunks, OUTPUT_PATH)

    # ── Statistics ────────────────────────────────────────────────────────────
    sizes = [c.metadata["token_count"] for c in chunks]
    print()
    print("── Chunk Statistics ─────────────────────────────────────")
    print(f"  Min tokens       : {min(sizes)}")
    print(f"  Max tokens       : {max(sizes)}")
    print(f"  Avg tokens       : {sum(sizes) // len(sizes)}")
    print(f"  With Illustrations : {sum(1 for c in chunks if c.metadata['has_illustration'])}")
    print(f"  With Explanations  : {sum(1 for c in chunks if c.metadata['has_explanation'])}")
    print(f"  With Exceptions    : {sum(1 for c in chunks if c.metadata['has_exception'])}")
    print(f"  With Provisos      : {sum(1 for c in chunks if c.metadata['has_proviso'])}")
    print(f"  With Definitions   : {sum(1 for c in chunks if c.metadata['has_definition'])}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
