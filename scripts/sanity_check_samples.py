import json
import random
from pathlib import Path

SOURCE_FILES = {
    'hadith': Path('processed/hadith/hadith_unified.jsonl'),
    'fatwa': Path('processed/fatwas/fatwas_unified.jsonl'),
    'tafsir': Path('processed/tafsir/tafsir_unified.jsonl'),
    'quran': Path('processed/quran/quran_unified.jsonl'),
}

HTML_PATTERN = '<[^>]+>'
MOJIBAKE_CANDIDATES = ['Ã', 'â', '�', '\ufffd']


def load_sample(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', encoding='utf-8') as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        raise ValueError(f'Empty file: {path}')
    raw = random.choice(lines)
    return json.loads(raw)


def is_clean_arabic(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # A basic Arabic script sanity check
    arabic_chars = sum(1 for ch in text if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF')
    return arabic_chars > 0


def contains_html(text: str) -> bool:
    import re
    return bool(re.search(HTML_PATTERN, text))


def contains_mojibake(text: str) -> bool:
    return any(ch in text for ch in MOJIBAKE_CANDIDATES)


for source_type, path in SOURCE_FILES.items():
    print(f'=== SAMPLE {source_type.upper()} ===')
    try:
        item = load_sample(path)
    except Exception as exc:
        print('ERROR:', exc)
        print()
        continue

    print('source_type:', item.get('source_type'))
    print('embedding_text non-empty:', bool(item.get('embedding_text')))
    print('embedding_text length:', len(item.get('embedding_text', '')))
    print('contains_html in embedding_text:', contains_html(item.get('embedding_text', '')))
    print('contains_mojibake in embedding_text:', contains_mojibake(item.get('embedding_text', '')))

    if source_type == 'hadith':
        text_field = item.get('clean_text') or item.get('text')
    elif source_type == 'fatwa':
        text_field = item.get('clean_text') or item.get('answer') or item.get('question')
    elif source_type == 'tafsir':
        text_field = item.get('clean_text') or item.get('text')
    elif source_type == 'quran':
        text_field = item.get('clean_text') or item.get('text')
    else:
        text_field = item.get('clean_text') or item.get('text')

    print('text field sample:', repr(text_field)[:500])
    print('text contains_html:', contains_html(text_field))
    print('text contains_mojibake:', contains_mojibake(text_field))
    print('text looks Arabic:', is_clean_arabic(text_field))
    print('metadata keys:', list(item.keys())[:20])
    print()
