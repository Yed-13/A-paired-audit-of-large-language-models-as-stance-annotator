#!/usr/bin/env python3
"""Normalize the official archive without modifying its contents or inventing labels."""
import csv
import hashlib
import io
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from pipeline import TARGETS, read_data, select_sample

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    folder = ROOT / 'data/private'
    archive = folder / 'stance-data-all-annotations.zip'
    z = ZipFile(archive)
    rows, members = [], {}
    for split, name in [('train', 'trainingdata-all-annotations.txt'),
                        ('test', 'testdata-taskA-all-annotations.txt')]:
        raw = z.read('data-all-annotations/' + name)
        members[name] = hashlib.sha256(raw).hexdigest()
        # Source bytes contain Windows-1252 curly quotation marks; decode strictly.
        source = list(csv.DictReader(io.StringIO(raw.decode('cp1252')), delimiter='\t'))
        for r in source:
            rows.append({'id': r['ID'], 'text': r['Tweet'], 'target': r['Target'],
                         'label': r['Stance'], 'split': split,
                         'opinion_towards': r['Opinion towards'], 'sentiment': r['Sentiment']})
    rng = random.Random(20260912)
    pilot_ids = set()
    for target in TARGETS:
        pool = sorted([r for r in rows if r['split'] == 'train' and r['target'] == target], key=lambda r:r['id'])
        pilot_ids.update(r['id'] for r in rng.sample(pool, 4))
    for r in rows:
        if r['id'] in pilot_ids:
            r['split'] = 'dev'
    csv_path = folder / 'semeval.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    checked = read_data(csv_path)
    sample = select_sample(checked, 120, 20260912)
    report = {
        'source_url': 'https://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip',
        'source_page': 'https://www.saifmohammad.com/WebPages/StanceDataset.htm',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'original_file_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'member_sha256': members, 'normalized_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'encoding': 'Windows-1252 -> UTF-8; strict decoding; text otherwise unchanged',
        'source_split_mapping': 'trainingdata -> train/dev; testdata-taskA -> test; Task B excluded',
        'normalization_and_dev_split': 'Four seeded training items per target reserved for dev; seed 20260912; original trial covers only two targets and is excluded',
        'status': 'local_research_use_checked_provider_processing_pending',
        'research_use_reviewed_by': 'Codex inspected official terms allowing research with attribution; source redistribution excluded',
        'provider_processing_review': 'PENDING provider selection and authorization; no text sent to models',
        'source_test_count': 1249, 'normalized_count': len(rows), 'main_sample_count': len(sample),
        'split_counts': dict(Counter(r['split'] for r in rows)),
        'main_sample_ids': [r['id'] for r in sample], 'pilot_ids': sorted(pilot_ids)}
    (folder / 'provenance.json').write_text(json.dumps(report,indent=2))
    (folder / 'sample-600.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
    tables = ROOT / 'outputs/dataset'
    tables.mkdir(parents=True,exist_ok=True)
    counts = []
    for split, subset in [('train', [r for r in rows if r['split']=='train']),
                          ('pilot', [r for r in rows if r['split']=='dev']),
                          ('test_pool',[r for r in rows if r['split']=='test']), ('main_sample',sample)]:
        for target in TARGETS:
            for label in ['FAVOR','AGAINST','NONE']:
                counts.append({'split':split,'target':target,'label':label,
                               'n':sum(r['target']==target and r['label']==label for r in subset)})
    with (tables/'composition.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','target','label','n']); w.writeheader();w.writerows(counts)
    (tables/'STATUS.json').write_text(json.dumps({'status':'REAL_SOURCE_DATA_DESCRIPTIVE_ONLY',
        'source_sha256':report['original_file_sha256'],'model_outputs_collected':0},indent=2))
    print(json.dumps({k:report[k] for k in ['source_test_count','main_sample_count','split_counts','original_file_sha256']},indent=2))


if __name__ == '__main__':
    prepare()
