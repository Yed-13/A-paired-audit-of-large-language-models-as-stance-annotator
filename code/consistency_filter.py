"""Post hoc descriptive unanimity filtering; no model calls or new labels."""
import csv
from collections import defaultdict
from pathlib import Path

def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['model'],row['condition'])].append(row)
    output=[]
    for (model,condition),rows in groups.items():
        complete=[x for x in rows if x['complete']=='True']
        unanimous=[x for x in complete if x['unanimous']=='True']
        other=[x for x in complete if x['unanimous']!='True']
        def wrong(xs):return sum(x['majority_correct']!='True' for x in xs)
        n,nu,no=len(complete),len(unanimous),len(other)
        assert n==nu+no
        output.append(dict(model=model,condition=condition,n_complete=n,n_unanimous=nu,
            n_nonunanimous=no,n_wrong_complete=wrong(complete),n_wrong_unanimous=wrong(unanimous),
            n_wrong_nonunanimous=wrong(other),retained_fraction=nu/n,
            error_complete=wrong(complete)/n,error_unanimous=wrong(unanimous)/nu,
            error_nonunanimous=wrong(other)/no if no else '',
            error_change_pp=100*(wrong(unanimous)/nu-wrong(complete)/n)))
    return output

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    with (root/'results/item_consistency.csv').open() as f:out=summarize(csv.DictReader(f))
    with (root/'results/consistency_filter.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('Wrote results/consistency_filter.csv (post hoc descriptive analysis).')
