"""Frozen adapter vs unchanged base and rules; no output repair or test-time fitting."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import statistics
from runtime import ROOT, load_adapter, generate
from data import load_splits, audit
from task import MODEL_ID, REVISION, digest, score, rule_baseline


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, __import__('math').ceil(len(ordered) * q) - 1))]


def evaluate(args):
    datasets = load_splits()
    manifest = audit(datasets)
    checkpoint = Path(args.adapter)
    training = json.loads((checkpoint.parent / 'training.json').read_text(encoding='utf-8'))
    if training['dataset']['dataset_hash'] != manifest['dataset_hash'] or training['base_revision'] != REVISION:
        raise ValueError('Dataset or base revision differs from the frozen training manifest')
    tokenizer, model, device = load_adapter(checkpoint, args.device, offline=not args.allow_download)
    challenge = [json.loads(line) for line in (ROOT / 'data/challenge.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if digest(challenge) != training['challenge_hash']:
        raise ValueError('Authored challenge changed after training was frozen')
    # Warm each mode on a training request; neither warm-up output is scored.
    for adapter in (False, True):
        with nullcontext() if adapter else model.disable_adapter():
            generate(tokenizer, model, [datasets['train'][0]['text']])
    results, predictions = {}, {}
    for name, cases in [('heldout_templates', datasets['test']), ('challenge', challenge)]:
        results[name] = {'rules': score(cases, [rule_baseline(c['text']) for c in cases])}
        predictions[name] = {}
        for variant in ('base', 'adapter'):
            outputs, batches = [], []
            with model.disable_adapter() if variant == 'base' else nullcontext():
                for i in range(0, len(cases), args.batch_size):
                    values, elapsed = generate(tokenizer, model, [c['text'] for c in cases[i:i + args.batch_size]])
                    outputs.extend(values)
                    batches.append(elapsed)
            result = score(cases, outputs)
            result['timing'] = {'batch_size': args.batch_size, 'total_generation_seconds': sum(batches),
                'amortized_seconds_per_case': sum(batches) / len(cases), 'batch_p50_seconds': statistics.median(batches),
                'batch_p95_seconds': percentile(batches, .95), 'scope': 'Warm batched generation only; not end-to-end request latency or a load test.'}
            results[name][variant] = result
            predictions[name][variant] = [{'id': case['id'], 'raw_output': value} for case, value in zip(cases, outputs)]
            print(json.dumps({'split': name, 'variant': variant, 'exact_match': result['exact_match'], 'schema_valid_rate': result['schema_valid_rate']}), flush=True)
    adapter_sha = hashlib.sha256((checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest()
    report = {'base_model': MODEL_ID, 'base_revision': REVISION, 'adapter_sha256': adapter_sha,
              'dataset': manifest, 'challenge_hash': digest(challenge), 'device': device,
              'training': training, 'evaluation': results, 'predictions': predictions,
              'scope': 'Synthetic English task, held-out wrapper templates/order IDs and eight authored challenge requests. No general customer-support accuracy, multilingual quality or production deployment claim.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='private/run-v1/best')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--output', default='reports/comparison.json')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16:
        parser.error('Batch size must be 1–16')
    evaluate(args)
