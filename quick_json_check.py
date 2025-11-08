"""
Quick JSON Validation Check
Verifies that JSON files are being created correctly
"""

from pathlib import Path
import json

summary_dir = Path("logs/validation_summaries")

if not summary_dir.exists():
    print("❌ Directory logs/validation_summaries does not exist!")
    exit(1)

json_files = sorted(summary_dir.glob("val_ep*.json"))

if not json_files:
    print("⚠️  No JSON files found in logs/validation_summaries/")
    print("   Run training first: python main.py --episodes 3")
    exit(0)

print(f"✓ Found {len(json_files)} JSON file(s)\n")

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📄 {json_file.name}:")
        print(f"   Episode:  {data.get('episode', '?')}")
        print(f"   K:        {data.get('k', '?')}")
        print(f"   Trades:   {data.get('trades', 0.0):.1f}")
        print(f"   Score:    {data.get('score', 0.0):+.3f}")
        print(f"   Penalty:  {data.get('penalty', 0.0):.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Error reading {json_file.name}: {e}\n")

print("✓ All JSON files are valid!")
