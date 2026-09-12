"""Post hoc training-majority baseline; no API calls or post text needed."""
import csv,json
from pathlib import Path
from collections import Counter,defaultdict
from analyze import metrics,LABELS
root=Path(__file__).resolve().parents[1]
counts=defaultdict(Counter)
for r in csv.DictReader((root/'data/split_labels.csv').open()):
    if r['split']=='train':counts[r['target']][r['label']]+=1
labels={t:c.most_common(1)[0][0] for t,c in counts.items()}
items=json.loads((root/'data/analysis_plan.json').read_text())['items']
result={'status':'post_hoc_training_majority_baseline','selected_using':'remaining training partition only; no test-label selection','labels':labels,'metrics':metrics([LABELS.index(i['label']) for i in items],[LABELS.index(labels[i['target']]) for i in items])}
print(json.dumps(result,indent=2))
