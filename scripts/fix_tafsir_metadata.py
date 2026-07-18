#!/usr/bin/env python3
"""Fix tafsir processed JSONL metadata using original tafsir JSON sources.

Creates corrected `surah`, `ayah_start`, `ayah_end`, `ayah_range` fields
and removes path-like `ayah` values. Writes updated .processed.jsonl files
in-place (atomic) and produces `processed/tafsir/metadata_fix_report.json`.
"""
from pathlib import Path
import json, glob, re, shutil, logging
from datetime import datetime

LOG = logging.getLogger('fix_tafsir_metadata')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ROOT = Path('.')
PROCESSED_DIR = ROOT / 'processed' / 'tafsir'
REPORT_PATH = PROCESSED_DIR / 'metadata_fix_report.json'
LOG_PATH = PROCESSED_DIR / 'metadata_fix.log'

def find_source_for_stem(stem: str):
    # search for a json file with matching stem anywhere in workspace
    matches = list(Path('.').rglob(f'{stem}.json'))
    if not matches:
        return None
    # prefer ones under /quran or containing 'tafsir' in path
    for p in matches:
        if 'quran' in str(p).lower() or 'tafsir' in str(p).lower():
            return p
    return matches[0]

def parse_path_ayah(ayah_val: str):
    # expecting patterns like 'data/0/groups/0/tafseer' or similar
    m = re.search(r'data/(\d+)/groups/(\d+)', ayah_val)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def process_file(proc_path: Path, report: dict):
    stem = proc_path.name.replace('.processed.jsonl','')
    src = find_source_for_stem(stem)
    if not src:
        LOG.warning('No source JSON found for %s', stem)
        report['files_skipped_no_source'].append(proc_path.name)
        return

    LOG.info('Using source %s for %s', src, proc_path.name)
    src_obj = json.loads(src.read_text(encoding='utf-8'))

    out_tmp = proc_path.with_suffix('.processed.jsonl.tmp')
    seen_surahs = set()
    skipped_recs = []
    written = 0
    examples = []

    with proc_path.open(encoding='utf-8') as fin, out_tmp.open('w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                LOG.exception('Invalid JSON in %s:%s', proc_path, lineno)
                report['errors'].append({'file': proc_path.name, 'line': lineno, 'error': str(e)})
                skipped_recs.append((lineno, 'invalid_json'))
                continue

            ayah_val = obj.get('ayah')
            fixed = False

            if isinstance(ayah_val, str) and 'data/' in ayah_val and 'groups/' in ayah_val:
                parsed = parse_path_ayah(ayah_val)
                if parsed is None:
                    skipped_recs.append((lineno, 'unparseable_path'))
                else:
                    data_idx, group_idx = parsed
                    try:
                        data_entry = src_obj['data'][data_idx]
                        surah = data_entry.get('chapter')
                        group = data_entry['groups'][group_idx]
                        ayah_start = group.get('start')
                        ayah_end = group.get('end')
                        # validation
                        if surah in (None, '', 0) or ayah_start is None or ayah_end is None:
                            skipped_recs.append((lineno, 'missing_values'))
                        elif not (isinstance(ayah_start, int) and isinstance(ayah_end, int) and ayah_start <= ayah_end):
                            skipped_recs.append((lineno, 'invalid_range'))
                        else:
                            # remove old path-like ayah
                            obj.pop('ayah', None)
                            # remove non-semantic metadata fields if any (keep core text fields)
                            for k in list(obj.keys()):
                                if k not in ('original_text','clean_text','normalized_text','embedding_text','source','sha256'):
                                    # keep 'source' and 'sha256'
                                    pass
                            # inject new metadata
                            obj['surah'] = surah
                            obj['ayah_start'] = ayah_start
                            obj['ayah_end'] = ayah_end
                            obj['ayah_range'] = f"{surah}:{ayah_start}-{ayah_end}"
                            seen_surahs.add(surah)
                            fixed = True
                            written += 1
                            if len(examples) < 10:
                                examples.append({'file': proc_path.name, 'line': lineno, 'surah': surah, 'ayah_start': ayah_start, 'ayah_end': ayah_end, 'ayah_range': obj['ayah_range']})
                    except Exception as e:
                        LOG.exception('Lookup failed for %s %s', proc_path.name, ayah_val)
                        skipped_recs.append((lineno, 'lookup_error'))
            else:
                # If ayah already in 'S:E' format like '1:1' or numeric, attempt to normalize into fields
                if isinstance(ayah_val, str) and re.match(r'^\d+:\d+(?:-\d+)?$', ayah_val):
                    # format 'surah:ayah' or 'surah: start-end'
                    m = re.match(r'^(\d+):(\d+)(?:-(\d+))?$', ayah_val)
                    if m:
                        surah = int(m.group(1))
                        a1 = int(m.group(2))
                        a2 = int(m.group(3)) if m.group(3) else a1
                        if a1 <= a2:
                            obj.pop('ayah', None)
                            obj['surah'] = surah
                            obj['ayah_start'] = a1
                            obj['ayah_end'] = a2
                            obj['ayah_range'] = f"{surah}:{a1}-{a2}"
                            seen_surahs.add(surah)
                            fixed = True
                            written += 1
                            if len(examples) < 10:
                                examples.append({'file': proc_path.name, 'line': lineno, 'surah': surah, 'ayah_start': a1, 'ayah_end': a2, 'ayah_range': obj['ayah_range']})
                else:
                    # leave other records untouched but count them
                    report['unchanged_records'] += 1

            # final validation: ensure no null metadata when we added them
            if fixed:
                if any(obj.get(k) is None for k in ('surah','ayah_start','ayah_end','ayah_range')):
                    skipped_recs.append((lineno, 'null_after_fix'))
                    report['errors'].append({'file': proc_path.name, 'line': lineno, 'error': 'null_after_fix'})
                    continue

            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # replace original file
    shutil.move(str(out_tmp), str(proc_path))
    report['files_processed'] += 1
    report['records_written'] += written
    report['surahs'].update(seen_surahs)
    report['examples'].extend(examples)
    if skipped_recs:
        report['skipped_records'][proc_path.name] = skipped_recs

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(LOG_PATH, encoding='utf-8'))

    report = {
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'files_found': 0,
        'files_processed': 0,
        'records_written': 0,
        'unchanged_records': 0,
        'surahs': set(),
        'multi_ayah_groups': 0,
        'examples': [],
        'skipped_records': {},
        'files_skipped_no_source': [],
        'errors': []
    }

    proc_files = sorted(PROCESSED_DIR.glob('*.processed.jsonl'))
    report['files_found'] = len(proc_files)

    for pf in proc_files:
        process_file(pf, report)

    # count multi-ayah groups from examples and from files
    # quick pass to count any record with ayah_start<ayah_end
    for pf in PROCESSED_DIR.glob('*.processed.jsonl'):
        with pf.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                a1 = o.get('ayah_start')
                a2 = o.get('ayah_end')
                if isinstance(a1, int) and isinstance(a2, int) and a2 > a1:
                    report['multi_ayah_groups'] += 1

    report['files_processed'] = int(report['files_processed'])
    report['records_written'] = int(report['records_written'])
    report['surahs'] = sorted(list(report['surahs']))
    report['finished_at'] = datetime.utcnow().isoformat() + 'Z'

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    LOG.info('Done. Report written to %s', REPORT_PATH)

if __name__ == '__main__':
    main()
