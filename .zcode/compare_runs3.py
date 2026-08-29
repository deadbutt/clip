# -*- coding: utf-8 -*-
"""GT from old segments.json (user markers embedded in text) vs machine labels of both runs."""
import json, re
from collections import Counter

OLD_DIR = r'D:\MOSS-Transcribe-Diarize\runs\5a25083f65c0'
NEW_DIR = r'D:\MOSS-Transcribe-Diarize\runs\2729f73660b3'
MARK = re.compile(r'（([^）]*)）')
SPK = re.compile(r'^[sS](\d+)$')

old_segs = json.load(open(OLD_DIR + r'\segments.json', encoding='utf-8'))
new_segs = json.load(open(NEW_DIR + r'\segments.json', encoding='utf-8'))

# ---- build GT portions from old segments' annotated text ----
# semantics: （sXX）labels the text BEFORE it. text after last marker inherits last speaker.
gt = []          # explicit portions
hallu = []       # flagged portions (无效/幻听)
unmarked = []    # no speaker info
for s in old_segs:
    parts = MARK.split(s['text'])
    plain = MARK.sub('', s['text']).strip()
    has_mark = any(SPK.match(p.strip()) for i, p in enumerate(parts) if i % 2 == 1)
    portions = []
    pending = ''
    last_spk = None
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            pending += chunk
        else:
            mk = chunk.strip()
            m = SPK.match(mk)
            if m:
                spk = 'S' + m.group(1).zfill(2)
                if pending.strip():
                    portions.append({'text': pending.strip(), 'speaker': spk})
                    pending = ''
                    last_spk = spk
            elif pending.strip():
                portions.append({'text': pending.strip(), 'speaker': None, 'note': mk})
                pending = ''
            elif portions:
                portions[-1].setdefault('note', mk)
    if pending.strip():
        portions.append({'text': pending.strip(), 'speaker': last_spk})
    if not portions:
        continue
    if len(portions) == 1:
        p = portions[0]
        p['start'], p['end'] = s['start'], s['end']
        (hallu if p.get('note') else (gt if p['speaker'] else unmarked)).append(p)
    else:
        # split time proportional to char length
        tot = sum(len(p['text']) for p in portions)
        t = s['start']
        for p in portions:
            d = (s['end'] - s['start']) * len(p['text']) / tot
            p['start'], p['end'] = t, t + d
            t += d
            (hallu if p.get('note') else (gt if p['speaker'] else unmarked)).append(p)

gt = [g for g in gt if g['end'] - g['start'] > 0.15]
print(f'GT explicit portions: {len(gt)}   hallucination-flagged: {len(hallu)}   unmarked: {len(unmarked)}')

def votes(segs, a, b):
    ov = Counter()
    for s in segs:
        o = min(s['end'], b) - max(s['start'], a)
        if o > 0:
            ov[s['speaker']] += o
    return ov

# crosstabs
xt_old, xt_new, xt_new_vs_old = Counter(), Counter(), Counter()
old_ok = new_ok = 0
new_mismatch = []
for g in gt:
    ov, nv = votes(old_segs, g['start'], g['end']), votes(new_segs, g['start'], g['end'])
    ol = ov.most_common(1)[0][0] if ov else 'NONE'
    nl = nv.most_common(1)[0][0] if nv else 'NONE'
    xt_old[(g['speaker'], ol)] += 1
    xt_new[(g['speaker'], nl)] += 1
    if ov and nv:
        xt_new_vs_old[(ol, nl)] += 1
    if ol == g['speaker']: old_ok += 1
    if nl == g['speaker']:
        new_ok += 1
    else:
        new_mismatch.append((g, nl, (nv.most_common(1)[0][1] / sum(nv.values())) if nv else 0))

n = len(gt)
print(f'\nOLD machine vs user GT:   {old_ok}/{n} = {old_ok/n*100:.1f}%')
for (a, b), c in sorted(xt_old.items()): print(f'   GT={a} old={b}: {c}')
print(f'\nNEW (separated) vs user GT: {new_ok}/{n} = {new_ok/n*100:.1f}%')
for (a, b), c in sorted(xt_new.items()): print(f'   GT={a} new={b}: {c}')
print(f'\nNEW vs OLD machine agreement: {sum(c for (a,b),c in xt_new_vs_old.items() if a==b)}/{sum(xt_new_vs_old.values())}')
for (a, b), c in sorted(xt_new_vs_old.items()): print(f'   old={a} new={b}: {c}')

print(f'\n--- NEW mismatches vs GT ({len(new_mismatch)}) ---')
for g, nl, conf in new_mismatch:
    print(f"[{g['start']:7.2f}-{g['end']:7.2f}] GT={g['speaker']} NEW={nl} conf={conf:.0%}  {g['text'][:60]!r}")

# ---- hallucination lines still present in new? ----
print('\n--- flagged (无效/幻听/弹幕/背景) portions and whether new run still has text there ---')
for h in hallu:
    nv = [s for s in new_segs if s['start'] < h['end'] and s['end'] > h['start']]
    cov = sum(min(s['end'], h['end']) - max(s['start'], h['start']) for s in nv)
    print(f"[{h['start']:7.2f}] note={h.get('note','')} newcov={cov/(h['end']-h['start']):.0%} spks={sorted(set(s['speaker'] for s in nv))}  GT:{h['text'][:40]!r}  NEW:{' | '.join(s['text'][:30] for s in nv)[:80]!r}")

# ---- real speech coverage: unmarked+gt portions missing in new run ----
print('\n--- speech present for user but new run has nothing/short there ---')
for g in gt + unmarked:
    nv = [s for s in new_segs if s['start'] < g['end'] and s['end'] > g['start']]
    cov = sum(min(s['end'], g['end']) - max(s['start'], g['start']) for s in nv)
    dur = g['end'] - g['start']
    if dur > 0.4 and cov / dur < 0.5:
        print(f"[{g['start']:7.2f}-{g['end']:7.2f}] cov={cov/dur:.0%}  GT:{g['text'][:60]!r}")

# ---- new-run over-split vs user segmentation ----
print('\n--- over-splitting: user kept 1 segment, new run has >=2 inside ---')
cnt = 0
for s in old_segs:
    inside = [x for x in new_segs if x['start'] >= s['start'] - 0.12 and x['end'] <= s['end'] + 0.12]
    if len(inside) >= 2:
        cnt += 1
        if cnt <= 20:
            print(f"[{s['start']:7.2f}-{s['end']:7.2f}] user: {MARK.sub('', s['text'])[:45]!r} -> new {len(inside)}: {' | '.join(x['text'][:20] for x in inside)}")
print(f'total over-split user segments: {cnt}')

# reverse
print('\n--- user SPLIT into multiple, new run merged into 1 ---')
cnt = 0
for x in new_segs:
    inside = [s for s in old_segs if s['start'] >= x['start'] - 0.12 and s['end'] <= x['end'] + 0.12]
    if len(inside) >= 2:
        cnt += 1
        if cnt <= 12:
            print(f"[{x['start']:7.2f}-{x['end']:7.2f}] new 1 seg: {x['text'][:45]!r} <- user {len(inside)}: {' | '.join(MARK.sub('', s['text'])[:20] for s in inside)}")
print(f'total new-merged: {cnt}')
