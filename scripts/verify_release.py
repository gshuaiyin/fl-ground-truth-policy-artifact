"""Verify the public allowlist, exclusions, JSON files, and SHA-256 inventory."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', 'replay'}
TEXT = {'.md', '.py', '.json', '.csv', '.txt', '.yml', '.yaml', '.svg', '.java', '.patch', '.lines', '.candidates', '.drawio'}
FORBIDDEN_PARTS = {'paper', 'external', 'data', 'raw', 'rankings', 'evidence'}
FORBIDDEN_SUFFIXES = {'.doc', '.docx', '.pt', '.pth', '.pkl', '.pickle', '.html', '.zip', '.gz', '.tar', '.log'}

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    errors = []
    inventory = files()
    manifest_file = ROOT/'PUBLIC_FILE_MANIFEST.json'
    if not manifest_file.is_file():
        errors.append('PUBLIC_FILE_MANIFEST.json is missing')
    else:
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        expected = set(manifest['files']) | {'PUBLIC_FILE_MANIFEST.json', 'SHA256SUMS.txt'}
        actual = {p.relative_to(ROOT).as_posix() for p in inventory}
        for rel in sorted(actual - expected): errors.append('File outside explicit public allowlist: '+rel)
        for rel in sorted(expected - actual): errors.append('Required allowlist file missing: '+rel)
    patterns = [
        re.compile('gh' + r'[posu]_[A-Za-z0-9]{20,}'),
        re.compile('github' + r'_pat_[A-Za-z0-9_]{20,}'),
        re.compile('BEGIN ' + r'(?:RSA |OPENSSH |EC )?PRIVATE KEY'),
        re.compile('[A-Za-z]:' + re.escape('\\Users\\'), re.I),
        re.compile(re.escape('/root/') + r'(?:autodl|workspace|home)', re.I),
        re.compile('connect' + r'\.[A-Za-z0-9.-]*gpuhub\.com', re.I),
    ]
    for p in inventory:
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink(): errors.append('Symlink not allowed: '+rel)
        if any(part in FORBIDDEN_PARTS for part in p.relative_to(ROOT).parts): errors.append('Excluded directory: '+rel)
        if p.suffix.lower() in FORBIDDEN_SUFFIXES: errors.append('Excluded file type: '+rel)
        if p.suffix.lower() == '.pdf' and not re.fullmatch(r'figures/v3\.5\.0/figure(?:[134567]|S[1-5])\.pdf', rel):
            errors.append('Only approved original figure PDFs are allowed: '+rel)
        if p.suffix.lower() in TEXT:
            content = p.read_text(encoding='utf-8-sig')
            if '\ufffd' in content: errors.append('Replacement character: '+rel)
            for pattern in patterns:
                if pattern.search(content): errors.append('Possible credential or private endpoint: '+rel); break
            if p.suffix.lower() == '.json':
                try: json.loads(content)
                except ValueError: errors.append('Invalid JSON: '+rel)
        if p.name in {'reviewer_b_full_normalized.json', 'audit_inventory.json', 'manual_audit_manifest.csv'}:
            errors.append('Original or individual review material excluded: '+rel)
    checksum_file = ROOT/'SHA256SUMS.txt'
    hashed = set()
    if not checksum_file.is_file(): errors.append('SHA256SUMS.txt is missing')
    else:
        for line in checksum_file.read_text(encoding='utf-8').splitlines():
            digest, rel = line.split('  ', 1)
            p = (ROOT/rel).resolve()
            if not p.is_relative_to(ROOT.resolve()): errors.append('Checksum path escapes package'); continue
            hashed.add(rel)
            if not p.is_file() or sha(p) != digest: errors.append('Checksum mismatch: '+rel)
        actual = {p.relative_to(ROOT).as_posix() for p in inventory if p != checksum_file}
        if hashed != actual: errors.append('Checksum inventory does not exactly cover public files')
    for stem in ['figure1', 'figure3', 'figure4', 'figure5', 'figure6', 'figure7']+[f'figureS{i}' for i in range(1,6)]:
        for extension in ['svg', 'pdf', 'png']:
            if not (ROOT/'figures/v3.5.0'/f'{stem}.{extension}').is_file(): errors.append('Missing original figure: '+stem+'.'+extension)
    if errors:
        print('Public release verification FAILED:')
        for error in errors: print('- '+error)
        return 1
    print(f'Public release verification passed: {len(inventory)} files, exact allowlist and checksums; excluded materials absent.')
    return 0

if __name__ == '__main__': sys.exit(main())
