"""Download pinned public files, resume interrupted bytes and verify LFS SHA-256."""
import hashlib
import json
import urllib.request
from pathlib import Path
from task import MODEL_ID, REVISION

FILES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'model.safetensors', 'merges.txt', 'vocab.json', 'README.md'}
ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / 'private' / ('base-' + REVISION[:12])


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download():
    url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Unexpected model revision')
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'files': {}}
    for item in metadata['siblings']:
        name = item['rfilename']
        if name not in FILES:
            continue
        path = BASE_PATH / name
        expected = item.get('lfs', {}).get('sha256')
        size = item['size']
        if not path.exists() or path.stat().st_size != size or (expected and sha256(path) != expected):
            partial = path.with_name(path.name + '.partial')
            for attempt in range(8):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset == size:
                        break
                    request = urllib.request.Request(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}',
                                                     headers={'Range': f'bytes={offset}-'} if offset else {})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        append = offset > 0 and response.status == 206
                        with partial.open('ab' if append else 'wb') as output:
                            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                                output.write(chunk)
                    if partial.stat().st_size == size:
                        break
                except OSError as exc:
                    print(f'Retrying {name}: {type(exc).__name__}', flush=True)
            if not partial.exists() or partial.stat().st_size != size or (expected and sha256(partial) != expected):
                raise ValueError(f'Incomplete or checksum-mismatched download: {name}')
            partial.replace(path)
        actual = sha256(path)
        manifest['files'][name] = {'sha256': actual, 'bytes': size}
        print('Verified', name, size, flush=True)
    (BASE_PATH / 'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return BASE_PATH


if __name__ == '__main__':
    print('Pinned model:', download())
