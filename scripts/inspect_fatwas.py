from pathlib import Path
import json
p1=Path('hadith-json-main/db/fatwas/fatwas.jsonl')
if not p1.exists():
    p1=Path('hadith-json-main/hadith-json-main/db/fatwas/fatwas.jsonl')
print('using', p1)
with p1.open(encoding='utf-8') as fh:
    for i,line in enumerate(fh):
        if i>9: break
        try:
            obj=json.loads(line)
            print('LINE',i,'keys=',list(obj.keys()))
            for k in ['question','title','answer','url','categories','related']:
                if k in obj:
                    print('  ',k, '=>', repr(obj.get(k))[:200])
        except Exception as e:
            print('err',e)

p2=Path('hadith-json-main/hadith-json-main/db/fatwas/Ibn_Othaymeen.json')
if p2.exists():
    raw=json.loads(p2.read_text(encoding='utf-8'))
    print('\nIbn_Othaymeen count=', len(raw))
    print('first keys=', list(raw[0].keys())[:20])
    print('sample question=>', repr(raw[0].get('question'))[:200])
    print('sample answer=>', repr(raw[0].get('answer'))[:200])
