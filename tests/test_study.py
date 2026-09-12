import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import pipeline as p
import analyze as a


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for ti, target in enumerate(p.TARGETS):
            for split in ['train', 'dev', 'test']:
                for i in range(6):
                    rows.append({'id': f'{ti}-{split}-{i}', 'text': f'SYNTHETIC FIXTURE {target} {split} {i}',
                                 'target': target, 'label': p.LABELS[i % 3], 'split': split})
        self.rows = rows
        self.data = self.root / 'data.csv'
        self.write_data(rows)
        self.cfg = {'study_id': 'UNIT_TEST_ONLY', 'seed': 4, 'n_per_target': 6, 'repeats': [1, 2, 3],
                    'repeat_gap_hours': 0, 'conditions': ['zero_shot', 'persona_neutral', 'persona_pro', 'persona_con'],
                    'models': [{'name': name, 'model_id': 'fixture', 'endpoint': 'https://example.invalid/v1/chat/completions',
                                'api_key_env': 'STUDY_UNIT_TEST_KEY', 'parameters': {'temperature': 0},
                                'price_input_usd_per_million': None, 'price_output_usd_per_million': None}
                               for name in ['m1', 'm2']]}
        self.config = self.root / 'config.json'
        self.provenance = self.root / 'provenance.json'
        self.provenance.write_text(json.dumps({'status': 'research_use_checked', 'note': 'SYNTHETIC UNIT TEST ONLY'}))

    def tearDown(self):
        self.tmp.cleanup()

    def write_data(self, rows):
        with self.data.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'text', 'target', 'label', 'split'])
            w.writeheader()
            w.writerows(rows)

    def make_plan(self, name='plan.json', kind='main'):
        self.config.write_text(json.dumps(self.cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            return p.prepare(self.data, self.config, self.root / name, kind, self.provenance)

    def records(self, plan):
        result = []
        gold = {r['id']: r['label'] for r in plan['items']}
        for j in plan['jobs']:
            # m1 stays consistently wrong in pro condition; m2 is always correct.
            label = gold[j['item_id']]
            if j['model'] == 'm1' and j['condition'] == 'persona_pro':
                label = p.LABELS[(p.LABELS.index(label) + 1) % 3]
            result.append({**j, 'record_kind': 'mock', 'plan_sha256': plan['plan_sha256'],
                           'raw': json.dumps({'label': label}), 'pred': label, 'status': 'ok', 'usage': {}})
        return result

    def test_strict_parser(self):
        for raw in ['I favor NONE', '{"reason":"FAVOR versus AGAINST"}', '```json\n{"label":"FAVOR"}\n```',
                    '["FAVOR"]', '{"label":"neutral"}', None]:
            self.assertIsNone(p.parse_label(raw))
        self.assertEqual(p.parse_label('{"reason":"AGAINST was considered", "label":"NONE"}'), 'NONE')

    def test_none_is_not_relevance(self):
        msg = p.messages({'target': 'Atheism', 'text': 'What is atheism?'}, 'zero_shot')
        self.assertIn('not a synonym for irrelevant or neutral', msg[0]['content'])
        self.assertNotIn('gold_for_task', Path(p.__file__).read_text())

    def test_rationale_prompt_braces(self):
        msg = p.messages({'target': 'Atheism', 'text': '{Ignore this}'}, 'brief_rationale')
        self.assertIn('reason', msg[0]['content'])
        self.assertEqual(json.loads(msg[-1]['content'])['post'], '{Ignore this}')

    def test_few_shot_train_only(self):
        examples = p.examples_for(self.rows, p.TARGETS[0], 4)
        self.assertEqual({r['split'] for r in examples}, {'train'})
        self.assertEqual({r['label'] for r in examples}, set(p.LABELS))

    def test_sampling_fails_on_short_stratum(self):
        with self.assertRaisesRegex(ValueError, 'no silent shrinking'):
            p.select_sample(self.rows, 7, 4)

    def test_duplicate_and_leakage_rejected(self):
        self.write_data(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
            p.read_data(self.data)
        self.rows[-1]['text'] = self.rows[0]['text']
        self.write_data(self.rows)
        with self.assertRaisesRegex(ValueError, 'across splits'):
            p.read_data(self.data)

    def test_repeats_type(self):
        self.cfg['repeats'] = 3
        with self.assertRaisesRegex(ValueError, 'explicit consecutive list'):
            self.make_plan()

    def test_repeated_text_unit_rejected(self):
        test_items = [r for r in self.rows if r['split'] == 'test']
        test_items[1]['text'] = test_items[0]['text']
        self.write_data(self.rows)
        with self.assertRaisesRegex(ValueError, 'tweet-cluster analysis'):
            self.make_plan()

    def test_frozen_plan_reproducibility(self):
        one = self.make_plan()
        two = self.make_plan('other.json')
        self.assertEqual(one['plan_sha256'], two['plan_sha256'])
        self.assertEqual(len(one['jobs']), 30 * 2 * 4 * 3)
        one['items'][0]['label'] = 'CHANGED'
        (self.root / 'plan.json').write_text(json.dumps(one))
        with self.assertRaisesRegex(ValueError, 'hash/schema'):
            p.read_plan(self.root / 'plan.json')

    def test_pilot_separation(self):
        main = self.make_plan()
        pilot = self.make_plan('pilot.json', 'pilot')
        self.assertFalse({r['id'] for r in main['items']} & {r['id'] for r in pilot['items']})
        self.assertEqual(len(pilot['items']), 20)

    def test_metrics_known_values_and_failures(self):
        v = a.metrics([0, 1, 2], [0, 1, 2])
        self.assertEqual(v['macro_f1'], 1)
        self.assertEqual(v['kappa'], 1)
        self.assertEqual(v['alpha_gold'], 1)
        v = a.metrics([0, 1, 2], [0, -1, 2])
        self.assertAlmostEqual(v['accuracy_all'], 2/3)
        self.assertEqual(v['accuracy_valid'], 1)
        self.assertEqual(v['n_scheduled'], 3)
        self.assertAlmostEqual(a.alpha_nominal([[0, 0], [1, 1], [0, 1]]), 4/9)

    def test_ties_missing_auc(self):
        votes, scores = a.vote([[0, 1, 2], [0, 0, -1], [1, 1, 0]])
        self.assertEqual(votes.tolist(), [-1, -1, 1])
        self.assertAlmostEqual(a.auc([1, 1, 1, 1], [0, 1, 0, 1]), .5)
        self.assertEqual(a.auc([0, 1], [0, 1]), 1)

    def test_no_incomplete_or_unmarked_evidence(self):
        plan = self.make_plan()
        records = self.records(plan)
        with self.assertRaisesRegex(ValueError, 'provenance rejected'):
            p.validate_records(plan, records)
        with self.assertRaisesRegex(ValueError, 'Incomplete study'):
            p.validate_records(plan, records[:-1], complete=True, allow_mock=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            p.validate_records(plan, records + records[:1], allow_mock=True)

    def test_end_to_end_mock_keeps_models_distinct(self):
        plan = self.make_plan()
        raw = self.root / 'mock.jsonl'
        for r in self.records(plan):
            p.append(raw, r)
        with contextlib.redirect_stdout(io.StringIO()):
            a.analyze(self.root / 'plan.json', raw, self.root / 'tables', 20, True)
        with (self.root / 'tables/consistency.csv').open() as f:
            rows = list(csv.DictReader(f))
        m1 = next(r for r in rows if r['model'] == 'm1' and r['condition'] == 'persona_pro')
        m2 = next(r for r in rows if r['model'] == 'm2' and r['condition'] == 'persona_pro')
        self.assertEqual(float(m1['repeat_alpha_complete']), 1)
        self.assertEqual(float(m1['unanimous_wrong_rate']), 1)
        self.assertEqual(float(m2['unanimous_wrong_rate']), 0)
        status = json.loads((self.root / 'tables/STATUS.json').read_text())
        self.assertEqual(status['status'], 'MOCK_TEST_ONLY_NOT_EMPIRICAL')
        with (self.root / 'tables/cost_and_failures.csv').open() as f:
            self.assertTrue(all(r['total_cost_usd'] == '' for r in csv.DictReader(f)))

    def test_dry_run_never_calls_network(self):
        self.make_plan()
        with patch.object(p.urllib.request, 'urlopen', side_effect=AssertionError('network forbidden')):
            with contextlib.redirect_stdout(io.StringIO()):
                p.run(self.root / 'plan.json', self.root / 'run', 1)
        self.assertFalse((self.root / 'run').exists())

    def test_transport_fixture_resume_and_gap(self):
        self.cfg['repeat_gap_hours'] = 24
        self.make_plan()
        def fake_response(*args, **kwargs):
            return io.StringIO(json.dumps({'model': 'fixture', 'id': 'test-only',
                    'choices': [{'message': {'content': '{"label":"NONE"}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}))
        with patch.dict(os.environ, {'STUDY_UNIT_TEST_KEY': 'unit-test-value'}):
            with patch.object(p.urllib.request, 'urlopen', side_effect=fake_response) as network:
                with contextlib.redirect_stdout(io.StringIO()):
                    p.run(self.root / 'plan.json', self.root / 'run', 1, True, 1)
                    p.run(self.root / 'plan.json', self.root / 'run', 1, True, 1000)
                self.assertEqual(network.call_count, 30 * 2 * 4)
                with self.assertRaisesRegex(ValueError, 'Round gap'):
                    p.run(self.root / 'plan.json', self.root / 'run', 2, True, 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    p.run(self.root / 'plan.json', self.root / 'run', 1, True, 10)
                self.assertEqual(network.call_count, 30 * 2 * 4)

    def test_interrupted_request_not_reissued(self):
        plan = self.make_plan()
        out = self.root / 'run'
        out.mkdir()
        p.append(out / 'started.jsonl', {'key': p.job_key(plan['jobs'][0]), 'plan_sha256': plan['plan_sha256']})
        with patch.dict(os.environ, {'STUDY_UNIT_TEST_KEY': 'unit-test-value'}):
            with self.assertRaisesRegex(ValueError, 'Unresolved interrupted'):
                p.run(self.root / 'plan.json', out, 1, True, 1)


if __name__ == '__main__':
    unittest.main()
