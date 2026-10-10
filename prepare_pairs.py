"""Render plain pairs with a local llama-server's actual template; no generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def escaped_line(text):
    return text.replace('\\', '\\\\').replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--max-tokens', type=int, default=480)
    args = parser.parse_args()
    raw = args.pairs.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    pairs = data['pairs']
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise SystemExit('Supply at least two matched pair objects.')
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('LLAMA_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key

    def post(path, body):
        request = Request(args.url.rstrip('/') + path, data=json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    records, lines = [], {'positive': [], 'negative': []}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or set(pair) != {'positive', 'negative'}:
            raise SystemExit(f'Pair {index}: expected exactly positive and negative strings.')
        record = {'index': index}
        for side in lines:
            text = pair[side]
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f'Pair {index}/{side}: empty or non-string prompt.')
            prompt = post('/apply-template', {
                'messages': [{'role': 'user', 'content': text}],
                'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False},
            })['prompt']
            ids = post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
            if not 1 <= len(ids) <= args.max_tokens:
                raise SystemExit(f'Pair {index}/{side}: {len(ids)} tokens exceeds the allowed range.')
            record[side] = {'prompt': prompt, 'tokens': ids}
            lines[side].append(escaped_line(prompt))
        if record['positive']['tokens'] == record['negative']['tokens']:
            raise SystemExit(f'Pair {index}: both sides tokenize identically.')
        records.append(record)
    args.out.mkdir(parents=True, exist_ok=True)
    for side, rows in lines.items():
        (args.out / (side + '.txt')).write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    (args.out / 'rendered.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'thinking': False,
        'max_tokens': max(len(r[s]['tokens']) for r in records for s in lines),
        'pairs': records,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(records)} pairs; inspect {args.out / "rendered.json"} before training.')


if __name__ == '__main__':
    main()
