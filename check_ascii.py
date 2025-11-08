"""Scan for non-ASCII characters in print statements"""
import re

def find_non_ascii_in_prints(filename):
    """Find non-ASCII characters in print statements"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    issues = []
    for i, line in enumerate(lines, 1):
        if 'print(' in line or 'print(f"' in line:
            # Check for non-ASCII characters
            for j, char in enumerate(line):
                if ord(char) > 127:
                    issues.append({
                        'line': i,
                        'char': char,
                        'ord': ord(char),
                        'hex': hex(ord(char)),
                        'position': j,
                        'text': line.strip()
                    })
    
    return issues

print("="*60)
print("SCANNING FOR NON-ASCII IN PRINT STATEMENTS")
print("="*60)

files_to_check = ['trainer.py', 'main.py', 'environment.py', 'agent.py']

total_issues = 0
for filename in files_to_check:
    try:
        issues = find_non_ascii_in_prints(filename)
        if issues:
            print(f"\n{filename}: {len(issues)} non-ASCII characters found")
            for issue in issues[:5]:  # Show first 5
                print(f"  Line {issue['line']}, pos {issue['position']}: "
                      f"'{issue['char']}' (U+{issue['hex'][2:].upper().zfill(4)})")
                print(f"    {issue['text'][:80]}...")
            if len(issues) > 5:
                print(f"    ... and {len(issues)-5} more")
            total_issues += len(issues)
        else:
            print(f"\n{filename}: OK (no non-ASCII in prints)")
    except FileNotFoundError:
        print(f"\n{filename}: Not found")

print("\n" + "="*60)
if total_issues == 0:
    print("ALL CLEAR! No non-ASCII characters in print statements")
else:
    print(f"FOUND {total_issues} non-ASCII characters that may cause issues")
print("="*60)
