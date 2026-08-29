# -*- coding: utf-8 -*-
"""Cross-tab: user （sXX） marks vs old-run labels vs new-run labels."""
import json, re, difflib
from collections import Counter

OLD_DIR = r'D:\MOSS-Transcribe-Diarize\runs\5a25083f65c0'
NEW_DIR = r'D:\MOSS-Transcribe-Diarize\runs\2729f73660b3'
MARK = re.compile(r'（([^）]*)）')
SPK = re.compile(r'^[sS](\d+)$')

def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())

def parse_srt(path):
    with open(path, encoding='utf-8-sig') as f:
        content = f.read()
    entries = []
    for b in re.split(r'\n\s*\n', content.strip()):
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 2: continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', lines[1].strip())
        if not m: continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600+g[1]*60+g[2]+g[3]/1000
        end = g[4]*3600+g[5]*60+g[6]+g[7]/1000
        entries.append({'idx': int(lines[0]), 'start': start, 'end': end,
                        'text': ' '.join(lines[2:]).strip()})
    return entries

old_entries = parse_srt(OLD_DIR + r'\subtitle.srt')
old_segs = json.load(open(OLD_DIR + r'\segments.json', encoding='utf-8'))
new_segs = json.load(open(NEW_DIR + r'\segments.json', encoding='utf-8'))

# only EXPLICIT user marks: （sXX） at end of a text chunk
explicit = []  # {start,end,gt,text}
for e in old_entries:
    parts = MARK.split(e['text'])
    pending = ''
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            pending += chunk
        else:
            mk = chunk.strip()
            m = SPK.match(mk)
            if m and pending.strip():
                explicit.append({'start': e['start'], 'end': e['end'],
                                 'gt': 'S' + m.group(1).zfill(2), 'text': pending.strip()})
                pending = ''
            elif not m and pending.strip():
                pending = ''
print(f'explicit user-marked chunks: {len(explicit)}')

def label_votes(segs, a, b):
    ov = Counter()
    for s in segs:
        o = min(s['end'], b) - max(s['start'], a)
        if o > 0: ov[s['speaker']] += o
    return ov

xt_old, xt_new = Counter(), Counter()
old_agree = 0
for x in explicit:
    o = label_votes(old_segs, x['start'], x['end'])
    n = label_votes(new_segs, x['start'], x['end'])
    ol = o.most_common(1)[0][0] if o else 'NONE'
    nl = n.most_common(1)[0][0] if n else 'NONE'
    xt_old[(x['gt'], ol)] += 1
    xt_new[(x['gt'], nl)] += 1
    if ol == x['gt']: old_agree += 1

print('\nuser-mark vs OLD-run label crosstab:')
for (g, l), c in sorted(xt_old.items()):
    print(f'  GT={g}  old={l}  n={c}')
print(f'old-run agreement with user marks: {old_agree}/{len(explicit)} = {old_agree/len(explicit)*100:.1f}%')

print('\nuser-mark vs NEW-run label crosstab:')
for (g, l), c in sorted(xt_new.items()):
    print(f'  GT={g}  new={l}  n={c}')
new_agree = sum(c for (g, l), c in xt_new.items() if g == l)
print(f'new-run agreement with user marks: {new_agree}/{len(explicit)} = {new_agree/len(explicit)*100:.1f}%')

# per-region: 30s buckets of user-mark agreement, old vs new
print('\nper-30s agreement (n / old-ok / new-ok):')
buckets = {}
for x in explicit:
    k = int(x['start'] // 30)
    b = buckets.setdefault(k, [0, 0, 0])
    b[0] += 1
    if label_votes(old_segs, x['start'], x['end']).most_common(1)[0][0] == x['gt']: b[1] += 1
    nv = label_votes(new_segs, x['start'], x['end'])
    if nv and nv.most_common(1)[0][0] == x['gt']: b[2] += 1
for k in sorted(buckets):
    n, o, nw = buckets[k]
    flag = ' <<<' if nw < o else ''
    print(f'  {k*30//60:02d}:{k*30%60:02d}-{(k+1)*30//60:02d}:{(k+1)*30%60:02d}  n={n:2d}  old={o:2d}  new={nw:2d}{flag}')

# same-speaker collapse detection in new run: user-marked chunks where new run has BOTH speakers mixed
print('\nnew-run mixed-speaker coverage on marked chunks:')
for x in explicit:
    nv = label_votes(new_segs, x['start'], x['end'])
    tot = sum(nv.values())
    if tot and len(nv) > 1 and nv.most_common(1)[0][1] / tot < 0.9:
        pass  # already captured by conf in previous run

# raw sample dump: new run labels 05:00-11:00 every seg (to see swap region)
print('\n--- new run segments 04:50-05:10 ---')
for s in new_segs:
    if 290 <= s['start'] <= 310:
        print(f"  [{s['start']:8.2f}-{s['end']:8.2f}] {s['speaker']} {s['text'][:55]!r}")
print('--- same window OLD run ---')
for s in old_segs:
    if 290 <= s['start'] <= 310:
        print(f"  [{s['start']:8.2f}-{s['end']:8.2f}] {s['speaker']} {s['text'][:55]!r}")
