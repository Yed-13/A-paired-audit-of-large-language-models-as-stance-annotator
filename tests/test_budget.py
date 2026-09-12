import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import budget

class BudgetTests(unittest.TestCase):
    def test_ceiling_release_and_unknown_cost(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'budget.json'
            p.write_text(json.dumps({'authorization':'user_confirmed_numeric_limit','limit_usd':1,'requests':{}}))
            budget.reserve(p,'first',.75)
            with self.assertRaisesRegex(ValueError,'ceiling'):
                budget.reserve(p,'second',.5)
            budget.settle(p,'first',.1)
            budget.reserve(p,'second',.5)
            budget.settle(p,'second',None)
            self.assertAlmostEqual(sum(r['accounted_usd'] for r in json.loads(p.read_text())['requests'].values()),.6)
            with self.assertRaisesRegex(ValueError,'already reserved'):
                budget.reserve(p,'second',.1)

    def test_missing_numeric_authorization(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'budget.json'
            p.write_text(json.dumps({'authorization':'pending_numeric_limit','limit_usd':None,'requests':{}}))
            with self.assertRaisesRegex(ValueError,'numeric user authorization'):
                budget.reserve(p,'first',.001)
