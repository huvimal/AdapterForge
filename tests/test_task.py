import copy
import json
import unittest
from data import build, audit
from task import parse_output, score, rule_baseline


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.request = 'Please refund ORD-1234. This is not urgent.'
        self.target = {'intent': 'refund', 'urgency': 'normal', 'order_id': 'ORD-1234'}

    def test_valid_output(self):
        self.assertTrue(parse_output(json.dumps(self.target), self.request)['grounded'])

    def test_markdown_not_repaired(self):
        self.assertFalse(parse_output('```json\n' + json.dumps(self.target) + '\n```', self.request)['json_valid'])

    def test_duplicate_keys_rejected(self):
        self.assertFalse(parse_output('{"intent":"refund","intent":"account"}', self.request)['json_valid'])

    def test_nonstandard_json_rejected(self):
        self.assertFalse(parse_output('{"order_id":NaN}', self.request)['json_valid'])

    def test_extra_fields_rejected(self):
        self.assertFalse(parse_output(json.dumps({**self.target, 'score': 1}), self.request)['schema_valid'])

    def test_invalid_category(self):
        self.assertFalse(parse_output(json.dumps({**self.target, 'intent': 'approve'}), self.request)['schema_valid'])

    def test_hallucinated_order_not_grounded(self):
        parsed = parse_output(json.dumps({**self.target, 'order_id': 'ORD-9999'}), self.request)
        self.assertTrue(parsed['schema_valid'])
        self.assertFalse(parsed['grounded'])

    def test_ambiguous_ids_not_grounded(self):
        self.assertFalse(parse_output(json.dumps(self.target), self.request + ' Or ORD-4567.')['grounded'])

    def test_substring_order_not_grounded(self):
        self.assertFalse(parse_output(json.dumps(self.target), 'ORD-12345')['grounded'])

    def test_invalid_schema_not_counted_as_correct_null(self):
        case = {'id': 'a', 'group': 'g', 'text': 'Hello', 'target': {**self.target, 'order_id': None}}
        result = score([case], ['not json'])
        self.assertEqual(result['field_accuracy']['order_id'], 0)
        self.assertEqual(result['exact_match'], 0)

    def test_perfect_case(self):
        case = {'id': 'a', 'group': 'g', 'text': self.request, 'target': self.target}
        result = score([case], [json.dumps(self.target)])
        self.assertEqual(result['exact_match'], 1)
        self.assertEqual(result['nonnull_field_micro']['f1'], 1)

    def test_rules_are_explicit_reference(self):
        self.assertEqual(json.loads(rule_baseline(self.request)), self.target)


class DataTests(unittest.TestCase):
    def test_split_counts_and_disjointness(self):
        self.assertEqual(audit(build())['counts'], {'train': 512, 'validation': 64, 'test': 64})

    def test_reproducibility(self):
        self.assertEqual(build(), build())

    def test_group_leak_rejected(self):
        datasets = build()
        datasets['test'][0]['group'] = datasets['train'][0]['group']
        with self.assertRaises(ValueError):
            audit(datasets)

    def test_duplicate_text_rejected(self):
        datasets = build()
        datasets['train'][1]['text'] = datasets['train'][0]['text']
        with self.assertRaises(ValueError):
            audit(datasets)

    def test_order_leak_rejected(self):
        datasets = build()
        train = next(r for r in datasets['train'] if r['target']['order_id'])
        test = next(r for r in datasets['test'] if r['target']['order_id'])
        old = test['target']['order_id']
        test['target']['order_id'] = train['target']['order_id']
        test['text'] = test['text'].replace(old, train['target']['order_id'])
        with self.assertRaises(ValueError):
            audit(datasets)
