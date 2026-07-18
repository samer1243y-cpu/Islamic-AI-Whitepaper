from pathlib import Path
import json
p=Path('hadith-json-main/quran/tafsir_14_Tafsir_Ibn_Kathir_arabic.json')
raw=json.loads(p.read_text(encoding='utf-8'))
print('top keys=', list(raw.keys())[:20])
if 'data' in raw and isinstance(raw['data'], list):
    print('data count=', len(raw['data']))
    first=raw['data'][0]
    if isinstance(first, dict):
        print('first keys=', list(first.keys())[:50])
        for k in list(first.keys())[:20]:
            print(k, '->', repr(first.get(k))[:200])
    else:
        print('first item not dict, type=', type(first))
else:
    print('no data list or data missing')
