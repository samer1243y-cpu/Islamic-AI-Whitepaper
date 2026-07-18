#!/usr/bin/env python3
"""
Process Quran tafsir JSON files deterministically (rule-based).

Produces per-record JSONL files in `processed/tafsir/` and a summary report.

Requirements followed:
- Keep `original_text` immutable
- Clean HTML using BeautifulSoup and safe regex
- Arabic normalization (remove diacritics, alef variants, tatweel, unify spaces)
- Create embedding-friendly text
- SHA256 of original_text
- Validation, logging, and skipping only invalid records

Usage: python scripts/process_tafsir.py
"""
from __future__ import annotations

import json
import re
import sys
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
import html
from datetime import datetime

try:
    from bs4 import BeautifulSoup
except Exception as e:  # pragma: no cover - runtime import guard
    print("BeautifulSoup (bs4) is required. Install with: pip install beautifulsoup4")
    raise


LOG = logging.getLogger("process_tafsir")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)
    # also log to console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(ch)


def find_tafsir_files(root: Path) -> List[Path]:
    """Recursively find JSON files likely to be tafsir files.

    Heuristic: filename contains 'tafsir' or is under a 'quran' folder.
    """
    matches: List[Path] = []
    for p in root.rglob("*.json"):
        name = p.name.lower()
        parts = [pp.lower() for pp in p.parts]
        if "tafsir" in name or any("quran" == pp for pp in parts) or "tafsir" in " ".join(parts):
            matches.append(p)
    return matches


def recursive_string_nodes(obj: Any, path: List[str] | None = None) -> Generator[Tuple[List[str], str], None, None]:
    """Yield (path_keys, string) for string leaf nodes in JSON object.

    Only yields strings longer than a small threshold to avoid metadata.
    """
    if path is None:
        path = []
    if isinstance(obj, str):
        text = obj.strip()
        if len(text) >= 20:
            yield (path, text)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from recursive_string_nodes(v, path + [str(k)])
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from recursive_string_nodes(v, path + [str(i)])
        return


def clean_html_and_noise(text: str) -> str:
    """Remove HTML tags, span/classes/styles and junk, preserving original textual content.

    Uses BeautifulSoup to strip tags safely, then unescapes entities and collapses whitespace.
    """
    if not text:
        return text
    soup = BeautifulSoup(text, "html.parser")
    # remove script/style elements just in case
    for s in soup(["script", "style"]):
        s.decompose()
    cleaned = soup.get_text(separator=" ")
    cleaned = html.unescape(cleaned)
    # Remove control characters and non-printable except Arabic letters and basic punctuation
    cleaned = re.sub(r"[\r\t\x0b\x0c\x0e-\x1f]", " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned


# Arabic normalization utilities
RE_TATWEEL = re.compile("\u0640+")
RE_ALEF_VARIANTS = re.compile("[\u0622\u0623\u0625\u0671]")
RE_ALEF_MAPPER = lambda m: "ا"
RE_ALIF_MAKSURA = re.compile("\u0649")  # ى -> ي
RE_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0670\u06D6-\u06ED\u0656-\u065F\u08D4-\u08E1]"
)


def normalize_arabic(text: str) -> str:
    """Deterministic Arabic normalization for search-only purposes.

    Rules applied:
    - Remove diacritics (tashkeel)
    - Map alef variants (آ أ إ ٱ) to ا
    - Map ى to ي
    - Remove tatweel (ـ)
    - Collapse spaces

    This does not remove words or change meanings beyond orthographic normalization.
    """
    if not text:
        return text
    s = text
    s = RE_TATWEEL.sub("", s)
    s = RE_ALEF_VARIANTS.sub(RE_ALEF_MAPPER, s)
    s = RE_ALIF_MAKSURA.sub("ي", s)
    s = RE_DIACRITICS.sub("", s)
    # unify hebrew/arabic spaces if any
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def prepare_embedding_text(normalized: str) -> str:
    """Create embedding_text: cleaned, normalized, minimal punctuation, preserve meaning.

    Rule-based: keep Arabic letters, Latin letters, digits, and basic separators. Remove excessive punctuation.
    """
    if not normalized:
        return normalized
    # Remove punctuation except Arabic/Latin letters and numbers and basic separators
    # Keep commas and periods as they can indicate sentence breaks
    cleaned = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFA-Za-z0-9\s\.,؛؟!]", " ", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_json_file(path: Path) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """Process a single JSON file and return (records_written, records_skipped, errors, examples).

    Examples: up to 5 before/after pairs for report.
    """
    LOG.info(f"Processing {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.exception(f"Failed to parse JSON file {path}: {e}")
        return (0, 0, 1, [])

    records: List[Dict[str, Any]] = []
    written = 0
    skipped = 0
    errors = 0
    examples: List[Dict[str, Any]] = []

    for node_path, original in recursive_string_nodes(raw):
        try:
            sha = sha256_of_text(original)
            clean = clean_html_and_noise(original)
            normalized = normalize_arabic(clean)
            embedding = prepare_embedding_text(normalized)

            if not clean:
                LOG.warning(f"Empty cleaned text for node {node_path} in {path}, skipping")
                skipped += 1
                continue

            rec = {
                "ayah": "/".join(node_path) if node_path else "",
                "surah": "",
                "original_text": original,
                "clean_text": clean,
                "normalized_text": normalized,
                "embedding_text": embedding,
                "source": "tafsir_ibn_kathir",
                "sha256": sha,
            }
            records.append(rec)
            written += 1
            if len(examples) < 5:
                examples.append({"path": "/".join(node_path), "before": original[:500], "after": clean[:500]})
        except Exception as e:
            LOG.exception(f"Error processing node {node_path} in {path}: {e}")
            errors += 1
            continue

    # write output JSONL
    out_dir = Path("processed/tafsir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (path.stem + ".processed.jsonl")
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        LOG.exception(f"Failed to write output for {path}: {e}")
        return (0, 0, 1, [])

    LOG.info(f"Wrote {written} records to {out_file}")
    return (written, skipped, errors, examples)


def generate_report(summary: Dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main(root: Path) -> int:
    start = datetime.utcnow().isoformat() + "Z"
    log_path = Path("processed/tafsir/process.log")
    setup_logging(log_path)

    files = find_tafsir_files(root)
    LOG.info(f"Found {len(files)} candidate JSON files for tafsir processing")

    total_written = 0
    total_skipped = 0
    total_errors = 0
    total_nodes = 0
    examples_all: List[Dict[str, Any]] = []
    processed_files = 0

    for f in files:
        w, s, e, examples = process_json_file(f)
        if w + s + e > 0:
            processed_files += 1
        total_written += w
        total_skipped += s
        total_errors += e
        total_nodes += w + s
        examples_all.extend([{"file": str(f), **ex} for ex in examples])

    report = {
        "source_project_root": str(root),
        "started_at": start,
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "files_found": len(files),
        "files_processed": processed_files,
        "records_written": total_written,
        "records_skipped": total_skipped,
        "errors": total_errors,
        "examples": examples_all[:20],
        "success_rate": None,
    }
    total_attempts = total_written + total_skipped + total_errors
    report["success_rate"] = (total_written / total_attempts) if total_attempts > 0 else None

    report_path = Path("processed/tafsir/report.json")
    generate_report(report, report_path)
    LOG.info(f"Processing complete. Report written to {report_path}")
    LOG.info(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    root_dir = Path.cwd()
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    rc = main(root_dir)
    sys.exit(rc)
