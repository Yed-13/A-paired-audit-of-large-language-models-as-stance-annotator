#!/usr/bin/env python3
"""Analyze complete v2 study records; requires numpy. Mock results are test-only."""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import LABELS, TARGETS, job_key, read_log, read_plan, validate_records


def alpha_nominal(matrix):
    """Units x raters, None or -1 missing; pairable-unit coincidence weighting."""
    disagree, marg = 0.0, Counter()
    for row in matrix:
        values = [v for v in row if v is not None and v != -1]
        n = len(values)
        if n < 2:
            continue
        c = Counter(values)
        marg.update(c)
        disagree += (n * n - sum(v * v for v in c.values())) / (n - 1)
    n = sum(marg.values())
    if n <= 1:
        return float('nan')
    expected = (n * n - sum(v * v for v in marg.values())) / (n * (n - 1))
    return 1 - disagree / n / expected if expected else float('nan')


def metrics(gold, pred):
    g, p = np.asarray(gold), np.asarray(pred)
    valid = p >= 0
    result = {'n_scheduled': len(g), 'n_valid': int(valid.sum()),
              'coverage': float(valid.mean()), 'accuracy_all': float((g == p).mean())}
    if not valid.any():
        return dict(result, accuracy_valid=np.nan, macro_f1=np.nan, kappa=np.nan, alpha_gold=np.nan)
    a, b = g[valid], p[valid]
    f1 = []
    for c in range(3):
        tp = ((a == c) & (b == c)).sum()
        denom = (a == c).sum() + (b == c).sum()
        f1.append(2 * tp / denom if denom else 0.0)
    pe = sum((a == c).mean() * (b == c).mean() for c in range(3))
    acc = (a == b).mean()
    return dict(result, accuracy_valid=float(acc), macro_f1=float(np.mean(f1)),
                kappa=float((acc - pe) / (1 - pe)) if pe < 1 else np.nan,
                alpha_gold=alpha_nominal(zip(a.tolist(), b.tolist())))


def auc(scores, outcomes):
    scores, outcomes = np.asarray(scores), np.asarray(outcomes)
    pos, neg = scores[outcomes == 1], scores[outcomes == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def vote(predictions):
    """Incomplete sequences and tied modes abstain (-1); never favor first label."""
    votes, scores = [], []
    for row in predictions:
        if any(v < 0 for v in row):
            votes.append(-1)
            scores.append(np.nan)
            continue
        counts = Counter(row)
        top = max(counts.values())
        winners = [k for k, n in counts.items() if n == top]
        votes.append(winners[0] if len(winners) == 1 else -1)
        scores.append(top / len(row))
    return np.array(votes), np.array(scores)


def ci(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    # Undefined replicates are counted, never silently presented as full evidence.
    if len(finite) < .9 * len(a) or not len(a):
        return {'ci_low': np.nan, 'ci_high': np.nan, 'bootstrap_defined': len(finite)}
    low, high = np.quantile(finite, [.025, .975])
    return {'ci_low': float(low), 'ci_high': float(high), 'bootstrap_defined': len(finite)}


def bootstrap_indices(targets, count, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(targets) == t) for t in sorted(set(targets))]
    return [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(count)]


def write_table(out, name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (out / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(plan_path, raw_path, output, n_boot=2000, allow_mock=False):
    if n_boot < 20:
        raise ValueError('Use at least 20 bootstrap draws; final reporting uses 2000 or more')
    plan = read_plan(plan_path)
    if plan['kind'] != 'main':
        raise ValueError('Pilot data cannot enter final study analysis')
    records = read_log(raw_path)
    validate_records(plan, records, complete=True, allow_mock=allow_mock)
    kinds = {r['record_kind'] for r in records}
    if len(kinds) != 1:
        raise ValueError('Mixed mock/live data prohibited')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Analysis destination must be empty; do not overwrite prior results')
    out.mkdir(parents=True, exist_ok=True)
    lookup = {job_key(r): r for r in records}
    items = plan['items']
    gold = np.array([LABELS.index(r['label']) for r in items])
    targets = [r['target'] for r in items]
    reps, conditions = plan['config']['repeats'], plan['config']['conditions']
    draws = bootstrap_indices(targets, n_boot, plan['config']['seed'])
    cubes, performance, stable, item_rows, cost, contrasts, prevalence = {}, [], [], [], [], [], []
    for model in plan['config']['models']:
        name = model['name']
        model_records = [r for r in records if r['model'] == name]
        known_cost, unknown_cost = 0.0, 0
        for r in model_records:
            usage = r.get('usage') or {}
            if isinstance(usage.get('cost'), (int,float)) and np.isfinite(usage['cost']) and usage['cost']>=0:
                known_cost += usage['cost']
                continue
            pin, pout = model.get('price_input_usd_per_million'), model.get('price_output_usd_per_million')
            if pin is None or pout is None or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
                unknown_cost += 1
            else:
                known_cost += (usage['prompt_tokens'] * pin + usage['completion_tokens'] * pout) / 1e6
        cost.append({'model': name, 'attempted_calls': len(model_records),
                     'api_errors': sum(r['status'] == 'api_error' for r in model_records),
                     'parse_errors': sum(r['status'] == 'parse_error' for r in model_records),
                     'known_cost_usd': known_cost, 'calls_with_unknown_cost': unknown_cost,
                     'total_cost_usd': known_cost if not unknown_cost else None})
        for condition in conditions:
            p = np.array([[LABELS.index(lookup[(name, condition, rep, r['id'])]['pred'])
                           if lookup[(name, condition, rep, r['id'])]['pred'] in LABELS else -1
                           for rep in reps] for r in items])
            cubes[name, condition] = p
            for col, rep in enumerate(reps):
                performance.append({'model': name, 'condition': condition, 'repeat': rep, **metrics(gold, p[:, col])})
            v, score = vote(p)
            complete = (p >= 0).all(axis=1)
            unanimous = complete & (score == 1)
            wrong = v != gold
            s = {'model': name, 'condition': condition, 'n_complete': int(complete.sum()),
                 'n_tied': int((complete & (v < 0)).sum()), 'n_unanimous': int(unanimous.sum()),
                 'repeat_alpha_complete': alpha_nominal(p[complete].tolist()),
                 'majority_accuracy_all': float((v == gold).mean()),
                 'unanimous_wrong_rate': float(wrong[unanimous].mean()) if unanimous.any() else np.nan,
                 'auc_majority_correct_complete': auc(score[complete], (v[complete] == gold[complete]).astype(int))}
            auc_draws, error_draws = [], []
            for ix in draws:
                use = ix[complete[ix]]
                auc_draws.append(auc(score[use], (v[use] == gold[use]).astype(int)))
                u = ix[unanimous[ix]]
                error_draws.append(wrong[u].mean() if len(u) else np.nan)
            s.update({'auc_' + k: val for k, val in ci(auc_draws).items()})
            s.update({'unanimous_wrong_' + k: val for k, val in ci(error_draws).items()})
            stable.append(s)
            for i, item in enumerate(items):
                item_rows.append({'model': name, 'condition': condition, 'item_id': item['id'], 'target': item['target'],
                                  'complete': bool(complete[i]), 'consistency': score[i],
                                  'majority_pred': LABELS[v[i]] if v[i] >= 0 else 'ABSTAIN',
                                  'majority_correct': bool(v[i] == gold[i]), 'unanimous': bool(unanimous[i])})
        # Accuracy and prevalence comparisons preserve paired item sequences.
        pairs = [('persona_pro', 'persona_neutral'), ('persona_con', 'persona_neutral'),
                 ('persona_pro', 'persona_con')]
        pairs += [(c, 'zero_shot') for c in ('few_shot', 'brief_rationale', 'persona_neutral')]
        for left, right in pairs:
            if left not in conditions or right not in conditions:
                continue
            a, b = cubes[name, left], cubes[name, right]
            deltas = (a == gold[:, None]).mean(axis=1) - (b == gold[:, None]).mean(axis=1)
            contrasts.append({'model': name, 'left': left, 'right': right,
                              'metric': 'accuracy_all_difference', 'estimate': float(deltas.mean()),
                              **ci([deltas[ix].mean() for ix in draws])})
            for target in TARGETS:
                ix = np.flatnonzero(np.asarray(targets) == target)
                # Complete paired sequences for label-distribution estimands, with denominator visible.
                paired = ((a[ix] >= 0) & (b[ix] >= 0)).all(axis=1)
                use = ix[paired]
                for label in range(3):
                    shift = (a[use] == label).mean(axis=1) - (b[use] == label).mean(axis=1)
                    boots = bootstrap_indices([target] * len(use), n_boot, plan['config']['seed']) if len(use) else []
                    prevalence.append({'model': name, 'left': left, 'right': right, 'target': target,
                                       'label': LABELS[label], 'n_paired_complete': len(use), 'n_target': len(ix),
                                       'share_difference': float(shift.mean()) if len(use) else np.nan,
                                       **ci([shift[j].mean() for j in boots])})
                # Descriptive support-minus-opposition sign change on exactly matched cases.
                if len(use):
                    ma = float(((a[use] == 0).sum() - (a[use] == 1).sum()) / a[use].size)
                    mb = float(((b[use] == 0).sum() - (b[use] == 1).sum()) / b[use].size)
                    prevalence.append({'model': name, 'left': left, 'right': right, 'target': target,
                                       'label': 'support_minus_opposition', 'n_paired_complete': len(use),
                                       'n_target': len(ix), 'left_margin': ma, 'right_margin': mb,
                                       'sign_reversal': bool(ma * mb < 0)})
    dataset = [{'target': target, 'label': label,
                'n': sum(r['target'] == target and r['label'] == label for r in items)}
               for target in TARGETS for label in LABELS]
    for file, rows in [('dataset.csv', dataset), ('performance_by_run.csv', performance),
                       ('consistency.csv', stable), ('item_consistency.csv', item_rows),
                       ('paired_accuracy_contrasts.csv', contrasts), ('prevalence_shifts.csv', prevalence),
                       ('cost_and_failures.csv', cost)]:
        write_table(out, file, rows)
    status = 'MOCK_TEST_ONLY_NOT_EMPIRICAL' if kinds == {'mock'} else 'OBSERVED_LIVE_RECORDS_NOT_EXTERNALLY_AUTHENTICATED'
    (out / 'STATUS.json').write_text(json.dumps({'status': status, 'plan_sha256': plan['plan_sha256'],
        'bootstrap_draws': n_boot, 'bootstrap_unit': 'item; stratified by target; all repetitions kept together',
        'intervals': 'pointwise percentile 95%; no multiplicity-adjusted significance claims',
        'coverage_policy': 'accuracy_all counts failures as incorrect; F1/kappa/alpha_gold use valid responses',
        'scope': 'selected targets, items, model configurations and observed rounds only'}, indent=2))
    print(status)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--allow-mock', action='store_true', help='Only for isolated software tests; never manuscript results')
    a = ap.parse_args()
    analyze(a.plan, a.raw, a.out, a.bootstrap, a.allow_mock)
