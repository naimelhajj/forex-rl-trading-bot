import json, glob, statistics as stats
from pathlib import Path

files = sorted(glob.glob("logs/validation_summaries/val_ep*.json"))
if not files:
    print("No validation summaries found.")
    raise SystemExit(0)

def get(d, k, default=None):
    return d.get(k, default)

rows = []
for f in files:
    with open(f, "r") as fh:
        j = json.load(fh)
    # Extract episode number from filename like "val_ep001.json"
    stem = Path(f).stem  # "val_ep001"
    ep_str = stem.split("_")[-1]  # "ep001"
    ep = int(ep_str.replace("ep", ""))  # Remove "ep" prefix, then convert to int
    rows.append({
        "ep": ep,
        "score": get(j, "score", None),
        "trades": get(j, "trades", None),
        "hold_rate": round(get(j, "hold_rate", 0.0), 3),
        "entropy": round(get(j, "action_entropy_bits", 0.0), 3),
        "avg_hold_len": round(get(j, "avg_hold_length", 0.0), 2),
        "max_hold": get(j, "hold_streak_max", 0),
        "switch_rate": round(get(j, "switch_rate", 0.0), 3),
        "long_ratio": round(get(j, "long_short", {}).get("long_ratio", 0.0), 3),
        "short_ratio": round(get(j, "long_short", {}).get("short_ratio", 0.0), 3),
    })

print(f"Found {len(rows)} validation summaries\n")
print("First 5 episodes:")
for r in rows[:5]:
    print(f"  Ep {r['ep']:>3}: score={r['score']:+.3f} | trades={r['trades']:>4} | "
          f"hold_rate={r['hold_rate']:.3f} | H(avg,max)=({r['avg_hold_len']:.2f},{r['max_hold']}) | "
          f"Hbits={r['entropy']:.3f} | switch={r['switch_rate']:.3f} | L/S=({r['long_ratio']:.3f}/{r['short_ratio']:.3f})")

def mean(x): 
    x = [v for v in x if isinstance(v,(int,float))]
    return (sum(x)/len(x)) if x else float('nan')

print("\nAverages:")
print(f"  hold_rate:       {mean([r['hold_rate'] for r in rows]):.3f}")
print(f"  avg_hold_length: {mean([r['avg_hold_len'] for r in rows]):.2f}")
print(f"  max_hold_streak: {max([r['max_hold'] for r in rows])}")
print(f"  action_entropy:  {mean([r['entropy'] for r in rows]):.3f} bits")
print(f"  switch_rate:     {mean([r['switch_rate'] for r in rows]):.3f}")
print(f"  long_ratio:      {mean([r['long_ratio'] for r in rows]):.3f}  (short={mean([r['short_ratio'] for r in rows]):.3f})")
