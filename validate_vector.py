"""Validate the small trained GGUF without loading model weights."""
import json
import struct
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def validate(path):
    with Path(path).open('rb') as stream:
        def read(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))[0]

        def string():
            return stream.read(read('Q')).decode('utf-8')

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                sub, count = read('I'), read('Q')
                return [value(sub) for _ in range(count)]
            return read({0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                         6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[kind])

        assert stream.read(4) == b'GGUF'
        version, count, keys = read('I'), read('Q'), read('Q')
        metadata = {}
        for _ in range(keys):
            name = string()
            metadata[name] = value(read('I'))
        tensors = []
        for _ in range(count):
            name, dimensions = string(), read('I')
            shape = [read('Q') for _ in range(dimensions)]
            kind, offset = read('I'), read('Q')
            tensors.append((name, shape, kind, offset))
        alignment = metadata.get('general.alignment', 32)
        start = (stream.tell() + alignment - 1) // alignment * alignment
        norms = []
        assert tensors, 'No control-vector tensors found'
        expected_shape = tensors[0][1]
        for index, (name, shape, kind, offset) in enumerate(tensors, 1):
            assert name == f'direction.{index}' and kind == 0 and len(shape) == 1
            assert shape == expected_shape, 'Inconsistent residual widths'
            stream.seek(start + offset)
            raw = stream.read(4 * shape[0])
            assert len(raw) == 4 * shape[0], 'Truncated tensor data'
            vector = np.frombuffer(raw, dtype='<f4')
            assert np.isfinite(vector).all()
            norm = float(np.linalg.norm(vector))
            assert abs(norm - 1) < 0.0001
            norms.append(norm)
    result = {'version': version, 'metadata': metadata, 'layers': count,
              'dimensions_per_layer': tensors[0][1][0], 'all_finite': True,
              'norm_min': min(norms), 'norm_max': max(norms)}
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('vector', type=Path)
    parser.add_argument('--layers', type=int, required=True, help='Expected number of exported vectors: model layer count minus one')
    parser.add_argument('--width', type=int, required=True, help='Model residual-stream width')
    args = parser.parse_args()
    result = validate(args.vector)
    assert result['layers'] == args.layers, result
    assert result['dimensions_per_layer'] == args.width, result
    assert result['metadata']['general.architecture'] == 'controlvector', result
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
