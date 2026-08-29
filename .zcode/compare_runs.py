# -*- coding: utf-8 -*-
"""Compare manual-calibrated SRT (ground truth) vs new vocal-separated run."""
import json, re, difflib
from collections import defaultdict, Counter

OLD_DIR = r'D:\MOSS-Transcribe-Diarize\runs\5a25083f65c0'
NEW_DIR = r'D:\MOSS-Transcribe-Diarize\runs\2729f73660b3'

MARK = re.compile(r'（([^）]*)）')
SPK = re.compile(r'^[sS](\d+)$')

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def parse_srt(path):
    with open(path, encoding='utf-8-sig') as f:
        content = f.read()
    entries = []
    for b in re.split(r'\n\s*\n', content.strip()):
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', lines[1].strip())
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        entries.append({'idx': int(lines[0]), 'start': start, 'end': end,
                        'text': '\n'.join(lines[2:]).replace('\n', ' ').strip()})
    return entries

def parse_portions(text):
    """Split entry text on （...） markers. （sXX） labels the text BEFORE it.
    Other markers are notes attached to preceding text (hallucination/TTS/etc)."""
    parts = MARK.split(text)
    portions = []
    pending = ''
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            pending += chunk
        else:
            mk = chunk.strip()
            if SPK.match(mk):
                if pending.strip():
                    portions.append({'text': pending.strip(), 'speaker': 'S' + SPK.match(mk).group(1).zfill(2), 'notes': []})
                    pending = ''
                elif portions:
                    portions[-1]['speaker'] = 'S' + SPK.match(mk).group(1).zfill(2)
            else:
                if pending.strip():
                    portions.append({'text': pending.strip(), 'speaker': None, 'notes': [mk]})
                    pending = ''
                elif portions:
                    portions[-1]['notes'].append(mk)
    if pending.strip():
        portions.append({'text': pending.strip(), 'speaker': None, 'notes': []})
    return portions

def fmt(t):
    m = int(t // 60); s = t - m * 60
    return f'{m:02d}:{s:06.3f}'

old_entries = parse_srt(OLD_DIR + r'\subtitle.srt')
old_segs = json.load(open(OLD_DIR + r'\segments.json', encoding='utf-8'))
new_segs = json.load(open(NEW_DIR + r'\segments.json', encoding='utf-8'))
new_entries = parse_srt(NEW_DIR + r'\subtitle.srt')

print(f'old srt entries={len(old_entries)}  old segs={len(old_segs)}  new segs={len(new_segs)}  new srt={len(new_entries)}')

# ---------- build ground truth portions with times ----------
# align manual entries to old segments (manual SRT was edited from old SRT, same 336 count)
gt = []  # {start,end,speaker,text,notes,strong}
for e in old_entries:
    portions = parse_portions(e['text'])
    # inherit speaker within entry
    last = None
    for p in portions:
        if p['speaker'] is None:
            p['speaker'] = last
        else:
            last = p['speaker']
    # time split: if 1 portion -> whole entry; else distribute by old segs or evenly
    if len(portions) == 1:
        portions[0]['start'], portions[0]['end'] = e['start'], e['end']
        gt.append(portions[0])
        continue
    # try aligning portions to old segments overlapping this entry by text
    osegs = [s for s in old_segs if s['start'] < e['end'] + 0.05 and s['end'] > e['start'] - 0.05]
    si = 0
    t0 = e['start']
    for pi, p in enumerate(portions):
        target = norm(p['text'])
        matched = []
        consumed = ''
        while si < len(osegs) and len(consumed) < max(len(target), 1):
            consumed += norm(osegs[si]['text'])
            matched.append(osegs[si])
            si += 1
        if matched and (target and target in norm(''.join(x['text'] for x in matched)) or
                        difflib.SequenceMatcher(None, target, consumed).ratio() > 0.6):
            p['start'] = matched[0]['start']
            p['end'] = matched[-1]['end']
            if p['speaker'] is None and len(matched) == 1:
                p['speaker'] = matched[0]['speaker']  # old-run label as weak GT
        else:
            # even split fallback
            span = (e['end'] - e['start']) / len(portions)
            p['start'] = e['start'] + span * pi
            p['end'] = e['start'] + span * (pi + 1)
        p.setdefault('start', e['start']); p.setdefault('end', e['end'])
        gt.append(p)

strong = [g for g in gt if g['speaker']]
weak_only = [g for g in gt if not g['speaker']]
print(f'GT portions={len(gt)}  strong(user-marked or aligned)={len(strong)}  unmarked-unaligned={len(weak_only)}')

# ---------- how good was the OLD run itself vs strong GT? ----------
old_hit = old_miss = 0
for g in strong:
    osegs = [s for s in old_segs if s['start'] < g['end'] and s['end'] > g['start']]
    if not osegs:
        continue
    ov = Counter()
    for s in osegs:
        o = min(s['end'], g['end']) - max(s['start'], g['start'])
        if o > 0:
            ov[s['speaker']] += o
    if ov and ov.most_common(1)[0][0] == g['speaker']:
        old_hit += 1
    else:
        old_miss += 1
print(f'OLD run vs user-corrected strong GT: agree={old_hit} disagree={old_miss} ({old_hit/(old_hit+old_miss)*100:.1f}%)')

# ---------- check S01/S02 identity consistency between runs ----------
probe = [
    (60.8, 62.5, "I'm at the College of Winterhold (Neuro)"),
    (436.07, 439.37, 'Nero, why do I feel latency problem (aunt)'),
    (751.7, 754.1, 'But I want talk with ya / No!'),
]
def segs_in(segs, a, b):
    return [s for s in segs if s['start'] < b and s['end'] > a]
print('\nlabel-identity probe (old vs new):')
for a, b, desc in probe:
    o = [(s['speaker'], s['text'][:40]) for s in segs_in(old_segs, a, b)]
    n = [(s['speaker'], s['text'][:40]) for s in segs_in(new_segs, a, b)]
    print(f'  [{fmt(a)}] {desc}\n    old={o}\n    new={n}')

# ---------- evaluate NEW run vs strong GT ----------
mismatch = []
correct = 0
for g in strong:
    nsegs = segs_in(new_segs, g['start'], g['end'])
    if not nsegs:
        mismatch.append((g, None, 0.0))
        continue
    ov = Counter()
    tot = 0.0
    for s in nsegs:
        o = min(s['end'], g['end']) - max(s['start'], g['start'])
        if o > 0:
            ov[s['speaker']] += o
            tot += o
    if tot <= 0:
        mismatch.append((g, None, 0.0))
        continue
    spk, o = ov.most_common(1)[0]
    if spk == g['speaker']:
        correct += 1
    else:
        mismatch.append((g, spk, o / tot))
acc = correct / len(strong) * 100
print(f'\nNEW run vs strong GT speaker accuracy: {correct}/{len(strong)} = {acc:.1f}%')
print(f'\n--- speaker MISMATCHES ({len(mismatch)}) ---')
for g, got, conf in mismatch:
    print(f"[{fmt(g['start'])}-{fmt(g['end'])}] GT={g['speaker']} NEW={got or 'NONE'} conf={conf:.0%} | GT: {g['text'][:60]!r} notes={g['notes']}")

# ---------- coverage: real speech missing in new run ----------
print('\n--- GT real-speech windows with NO new segment (missing speech) ---')
for g in gt:
    if any('无效' in n or '幻听' in n for n in g['notes']):
        continue
    nsegs = segs_in(new_segs, g['start'], g['end'])
    cov = sum(min(s['end'], g['end']) - max(s['start'], g['start']) for s in nsegs)
    dur = g['end'] - g['start']
    if dur > 0.3 and cov / dur < 0.4:
        print(f"[{fmt(g['start'])}-{fmt(g['end'])}] cov={cov/dur:.0%} spk={g['speaker']} notes={g['notes']} | {g['text'][:70]!r}")

# ---------- hallucination/TTS lines: does new run still have them ----------
print('\n--- flagged lines (无效/幻听/弹幕打赏/背景) in manual vs new run ---')
for g in gt:
    if not g['notes']:
        continue
    nsegs = segs_in(new_segs, g['start'], g['end'])
    ntxt = ' / '.join(s['text'][:50] for s in nsegs)
    nspk = set(s['speaker'] for s in nsegs)
    print(f"[{fmt(g['start'])}] notes={g['notes']} spk={g['speaker']}->new={sorted(nspk)}\n    GT: {g['text'][:70]!r}\n    NEW: {ntxt[:120]!r}")

# ---------- text differences old vs new (aligned by time overlap) ----------
print('\n--- text differences (old-manual vs new, per overlapping pair) ---')
diff_count = 0
used = set()
for o in old_entries:
    best, bo = None, 0
    for i, n in enumerate(new_entries):
        if i in used:
            continue
        ov = min(n['end'], o['end']) - max(n['start'], o['start'])
        if ov > bo:
            bo, best = ov, i
    if best is None or bo <= 0:
        continue
    used.add(best)
    n = new_entries[best]
    a, b = norm(o['text']), norm(n['text'])
    if a != b and difflib.SequenceMatcher(None, a, b).ratio() < 0.98:
        diff_count += 1
        sm = difflib.SequenceMatcher(None, a, b)
        ops = [x for x in sm.get_opcodes() if x[0] != 'equal']
        tag = '; '.join(f'{x[0]}:{o["text"][min(x[1],len(o["text"])):min(x[2],len(o["text"]))] or "∅"}->{n["text"][min(x[3],len(n["text"])):min(x[4],len(n["text"]))] or "∅"}' for x in ops[:4])
        print(f"[{fmt(o['start'])}] {tag[:150]}")
print(f'total text-diff entries: {diff_count}')

# ---------- micro-split analysis ----------
print('\n--- over-splitting in new run (consecutive new segs that user kept merged) ---')
splits = 0
for o in old_entries:
    inside = [s for s in new_segs if s['start'] >= o['start'] - 0.15 and s['end'] <= o['end'] + 0.15]
    if len(inside) >= 2:
        splits += 1
        if splits <= 15:
            print(f"[{fmt(o['start'])}-{fmt(o['end'])}] user=1 entry -> new={len(inside)} segs: {' | '.join(s['text'][:25] for s in inside)}")
print(f'user-merged-but-new-split entries: {splits}')

# reverse: user split but new merged
print('\n--- user SPLIT but new run MERGED ---')
for s in old_segs:
    inside = [e for e in old_entries if e['start'] >= s['start'] - 0.15 and e['end'] <= s['end'] + 0.15]
for e in old_entries:
    pass
merged = 0
for ns in new_segs:
    oe = [o for o in old_entries if o['start'] >= ns['start'] - 0.15 and o['end'] <= ns['end'] + 0.15]
    if len(oe) >= 2:
        merged += 1
        if merged <= 10:
            print(f"[{fmt(ns['start'])}-{fmt(ns['end'])}] new=1 seg ({ns['text'][:40]!r}) <- user had {len(oe)} entries: {' | '.join(o['text'][:30] for o in oe)}")
print(f'new-merged-but-user-split: {merged}')
