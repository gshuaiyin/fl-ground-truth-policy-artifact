"""Compare replayed statistical tables with the frozen original figure tables."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))

def equal(left, right):
    if left == right: return True
    try: return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    except (ValueError, TypeError): return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=ROOT/'replay/figures')
    args = parser.parse_args()
    records = []
    errors = []
    for original in sorted((ROOT/'figures/v3.5.0').glob('*.csv')):
        replay = args.replay/original.name
        if not replay.is_file(): continue
        left, right = read(original), read(replay)
        if len(left) != len(right): errors.append(original.name+': different row counts'); continue
        for index, (a, b) in enumerate(zip(left, right), 1):
            if a.keys() != b.keys(): errors.append(f'{original.name}: columns differ'); break
            for key in a:
                if not equal(a[key], b[key]): errors.append(f'{original.name}: row {index}, column {key} differs')
        records.append({'file':original.name, 'rows':len(left)})
    if len(records) < 8: errors.append('Expected at least 8 matched statistical tables')
    report = {'status':'FAIL' if errors else 'PASS', 'tables':records, 'errors':errors,
              'tolerance':{'relative':1e-10,'absolute':1e-12},
              'scope':'Frozen figure table values; not upstream input availability or experiment rerun'}
    args.replay.mkdir(parents=True, exist_ok=True)
    (args.replay/'TABLE_COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return int(bool(errors))

if __name__ == '__main__': raise SystemExit(main())
