import json, sys, collections

def load(job):
    with open(f"runs/{job}/segments.json", encoding="utf-8") as f:
        segs = json.load(f)
    return segs

def analyze(name, segs, show_minor=6):
    print(f"\n===== {name} ({len(segs)} segs, {segs[-1]['end']:.0f}s total) =====")
    per = collections.defaultdict(lambda: {"t": 0.0, "n": 0})
    for s in segs:
        p = per[s["speaker"]]
        p["t"] += s["end"] - s["start"]
        p["n"] += 1
    total = sum(p["t"] for p in per.values())
    order = sorted(per.items(), key=lambda kv: -kv[1]["t"])
    for spk, p in order:
        print(f"  {spk}: {p['t']:7.1f}s ({100*p['t']/total:5.1f}%)  {p['n']:4d} segs  mean {p['t']/p['n']:.1f}s")

    # switching pattern: consecutive same-speaker runs
    runs, cur, curlen = [], segs[0]["speaker"], 0.0
    for s in segs:
        if s["speaker"] == cur:
            curlen += s["end"] - s["start"]
        else:
            runs.append((cur, curlen)); cur, curlen = s["speaker"], s["end"] - s["start"]
    runs.append((cur, curlen))
    switches = len(runs) - 1
    print(f"  speaker switches: {switches} ({switches/ (segs[-1]['end']/60):.1f} per min)")
    print(f"  longest continuous block per speaker:")
    for spk in per:
        best = max((l for sp, l in runs if sp == spk), default=0)
        print(f"    {spk}: {best:.1f}s")

    # minor speakers: sample texts + where they live on the timeline
    for spk, p in order[1:3]:
        ss = [s for s in segs if s["speaker"] == spk]
        if not ss:
            continue
        span = (min(s["start"] for s in ss), max(s["end"] for s in ss))
        print(f"  --- {spk} sample ({span[0]:.0f}s..{span[1]:.0f}s) ---")
        for s in ss[:show_minor]:
            print(f"    [{s['start']:7.1f}] {s['text'][:70]!r}")

for job, name in [("c53bb7b2325a", "bajiru 歌切 21min 演唱+直播背景"),
                  ("5256902b5257", "EWC vlog 混剪")]:
    analyze(name, load(job))
