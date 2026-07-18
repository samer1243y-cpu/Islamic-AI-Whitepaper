import sys
from pathlib import Path

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline import gather_source_files, _debug_inspect_file, setup_logger

logger = setup_logger(True)
files = gather_source_files()

cats = {'fatwas':[], 'tafsir':[], 'quran':[], 'hadith':[], 'other':[]}
for p in files:
    parts=[x.lower() for x in Path(p).parts]
    if any('fatwa' in x for x in parts): cats['fatwas'].append(p)
    elif any('tafsir' in x for x in parts): cats['tafsir'].append(p)
    elif any('quran' in x for x in parts) or Path(p).suffix.lower()=='.xml': cats['quran'].append(p)
    elif any('by_book' in x or 'by_chapter' in x or 'hadith' in x for x in parts): cats['hadith'].append(p)
    else: cats['other'].append(p)

for k in cats:
    print(f"== {k} ({len(cats[k])}) ==")
    for sample in cats[k][:5]:
        print('  ', sample)
        _debug_inspect_file(sample, logger)
