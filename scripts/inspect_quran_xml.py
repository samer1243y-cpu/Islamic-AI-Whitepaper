from pathlib import Path
import xml.etree.ElementTree as ET
p=Path('hadith-json-main/quran/quran-uthmani.xml')
if not p.exists():
    p=Path('hadith-json-main/hadith-json-main/quran/quran-uthmani.xml')
print('using', p)
root=ET.parse(p).getroot()
print('root tag', root.tag)
# print first sura tag and attributes and first children
suras = root.findall('.//sura') or root.findall('sura')
print('suras found', len(suras))
if suras:
    s=suras[0]
    print('sura tag', s.tag, 'attrib', s.attrib)
    children=list(s)[:5]
    for c in children:
        print(' child tag', c.tag, 'attrib', c.attrib, 'text sample', repr((c.text or '')[:200]))
    # print deeper for first aya
    for aya in s.iter():
        if 'aya' in aya.tag.lower() or 'verse' in aya.tag.lower() or 'ayah' in aya.tag.lower():
            print('found aya tag', aya.tag, 'text sample', repr((aya.text or '')[:200]))
            break
else:
    # print first 2000 chars of file
    print(p.read_text(encoding='utf-8')[:2000])
