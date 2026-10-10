"""Patch only the pinned standalone trainer; leave inference/server code untouched."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path, help='Root of the pinned llama.cpp checkout')
    parser.add_argument('--check', action='store_true', help='Validate the patch without writing')
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name('trainer-patch.json').read_text(encoding='utf-8'))
    target = args.source / spec['file']
    original = target.read_text(encoding='utf-8')
    digest = hashlib.sha256(original.encode()).hexdigest()
    if digest == spec['patched_sha256']:
        print('Trainer already matches this patch.'); return
    if digest != spec['original_sha256']:
        raise SystemExit('Source differs from the pinned pristine trainer. Refusing to overwrite it.')
    updated = original
    for before, after in spec['replacements']:
        if updated.count(before) != 1:
            raise SystemExit('Patch anchor mismatch; no changes written.')
        updated = updated.replace(before, after)
    assert hashlib.sha256(updated.encode()).hexdigest() == spec['patched_sha256']
    if not args.check:
        target.write_text(updated, encoding='utf-8', newline='\n')
    print('Patch checked.' if args.check else 'Patched standalone trainer only.')


if __name__ == '__main__':
    main()
