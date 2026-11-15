"""Self-authored English task data. Template groups and order IDs stay split-local."""
import json
import random
from pathlib import Path
from task import digest, parse_output

ROOT = Path(__file__).resolve().parent
TEMPLATES = {
    'train': [
        ('t01', 'Hello support. {issue} {order} {urgency}'),
        ('t02', 'Support ticket: {issue} {urgency} {order}'),
        ('t03', 'Please help: {issue} {order} {urgency}'),
        ('t04', 'My message is simple. {issue} {urgency} {order}'),
        ('t05', 'Dear team, {issue} {order} {urgency}'),
        ('t06', '{urgency} Here is the problem: {issue} {order}'),
        ('t07', '{order} Request details: {issue} {urgency}'),
        ('t08', 'Could you assist me? {issue} {order} {urgency}'),
    ],
    'validation': [('v01', 'For the help desk to review: {order} {issue} {urgency}'),
                   ('v02', 'I am contacting customer care because {issue} {urgency} {order}')],
    'test': [('e01', 'A new customer enquiry follows. {urgency} {order} {issue}'),
             ('e02', 'Would someone on the service desk handle this? {issue} {order} {urgency}')],
}
ISSUES = {
    'refund': ['I want a refund.', 'Please give my money back.', 'I need to return the item.', 'I want to return this purchase.'],
    'delivery': ['Where is my shipment?', 'Please check the delivery status.', 'I need tracking for my package.', 'My parcel has not arrived.'],
    'account': ['I cannot log in.', 'I forgot my password.', 'My account access is blocked.', 'I cannot sign in.'],
    'other': ['What are your opening hours?', 'Do you offer gift wrapping?', 'Which colors are available?', 'Where is the store located?'],
}
NORMAL = ['This is not urgent.', 'There is no urgency.', 'A regular response is fine.', 'No immediate reply is needed.']
URGENT = ['This is urgent.', 'I need an immediate response.', 'Please handle this right away.', 'Urgent help requested.']


def build():
    rng = random.Random(20261115)
    datasets = {}
    for split, templates in TEMPLATES.items():
        cases = []
        base = {'train': 1000, 'validation': 5000, 'test': 8000}[split]
        for group, template in templates:
            for intent, phrases in ISSUES.items():
                for urgency in ('normal', 'urgent'):
                    for has_order in (False, True):
                        # Two variants per combination per group.
                        for variant in range(4 if split == 'train' else 2):
                            index = len(cases)
                            order_id = f'ORD-{base + index:04}' if has_order else None
                            phrase = phrases[(variant + index // 8) % len(phrases)]
                            order = f'Reference: {order_id}.' if order_id else 'No order reference is available.'
                            urgency_text = rng.choice(URGENT if urgency == 'urgent' else NORMAL)
                            text = template.format(issue=phrase, order=order, urgency=urgency_text)
                            cases.append({'id': f'{split}-{index:04}', 'group': group, 'text': text,
                                          'target': {'intent': intent, 'urgency': urgency, 'order_id': order_id}})
        rng.shuffle(cases)
        datasets[split] = cases
    return datasets


def audit(datasets):
    names = ('train', 'validation', 'test')
    groups, texts, orders = {}, {}, {}
    for name in names:
        rows = datasets[name]
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Empty split or duplicate case IDs')
        groups[name] = {r['group'] for r in rows}
        texts[name] = {r['text'] for r in rows}
        if len(texts[name]) != len(rows):
            raise ValueError('Duplicate requests within a split')
        orders[name] = {r['target']['order_id'] for r in rows if r['target']['order_id'] is not None}
        for row in rows:
            parsed = parse_output(json.dumps(row['target']), row['text'])
            if not parsed['schema_valid'] or not parsed['grounded']:
                raise ValueError('Invalid authored reference')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] & groups[b] or texts[a] & texts[b] or orders[a] & orders[b]:
                raise ValueError('Group, exact request or order ID crosses splits')
    return {'counts': {n: len(datasets[n]) for n in names}, 'group_counts': {n: len(groups[n]) for n in names},
            'split_hashes': {n: digest(datasets[n]) for n in names}, 'dataset_hash': digest(datasets),
            'scope': 'Self-authored synthetic English templates. Template groups and order IDs held out; task vocabulary is shared.'}


def load_splits():
    return {name: [json.loads(line) for line in (ROOT / 'data' / f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            for name in ('train', 'validation', 'test')}


if __name__ == '__main__':
    datasets = build()
    manifest = audit(datasets)
    path = ROOT / 'data'
    path.mkdir(exist_ok=True)
    for name, cases in datasets.items():
        (path / f'{name}.jsonl').write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in cases) + '\n', encoding='utf-8')
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts']))
