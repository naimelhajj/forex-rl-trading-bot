import json, glob, math
from pathlib import Path

def compute_from_counts(counts):
    total = sum(counts.values()) or 1
    probs = [(c/total) for c in counts.values()]
    entropy = -sum(p*math.log2(p) for p in probs if p>0)
    long = counts.get("LONG", 0)
    short = counts.get("SHORT", 0)
    ls = long + short
    return {
        "action_entropy_bits": entropy,
        "long_short": {
            "long": int(long),
            "short": int(short),
            "long_ratio": (long/ls) if ls else 0.0,
            "short_ratio": (short/ls) if ls else 0.0,
        }
    }

paths = sorted(glob.glob("logs/validation_summaries/val_ep*.json"))
updated = 0
for p in paths:
    with open(p, "r") as fh:
        j = json.load(fh)
    counts = j.get("actions")
    changed = False
    if isinstance(counts, dict):
        add = compute_from_counts(counts)
        # only add if missing
        for k, v in add.items():
            if k not in j:
                j[k] = v
                changed = True
    if changed:
        with open(p, "w") as fh:
            json.dump(j, fh, indent=2)
        updated += 1

print(f"Updated {updated} files.")
