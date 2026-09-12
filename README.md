# Consistent labels, shifting conclusions: a paired audit of large language models as stance annotators

Experimental data, prompts, code and results for a paired audit of large language models used as stance annotators.

## Design and data

- 600 sampled post–target pairs from SemEval-2016 Task 6, with 120 items per target.
- Two hosted models, four prompt conditions and three rounds.
- 14,400 attempted annotations: 14,392 valid labels and eight retained API failures.
- Temperature zero, with at least 30 minutes between completed rounds.

`data/` contains a text-free analytical plan, all recorded outcomes, exact prompt templates and source split labels. `results/` contains the seven primary CSV tables, a post hoc unanimity-filter table, the training-majority baseline and generated plots. `code/` and `tests/` support offline reproduction. No source post text, full request/response bodies, credentials or private records are included.

## Reproduce offline

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/analyze.py --plan data/analysis_plan.json --raw data/predictions.jsonl --out reproduced --bootstrap 2000
python code/baseline.py > reproduced/descriptive_baseline.json
python code/consistency_filter.py
python code/experiment_figures.py
python -m unittest discover -s tests -q
```

The seven primary CSV tables under `reproduced/` should match their counterparts under `results/`. The filtering script writes `results/consistency_filter.csv`; the figure script writes `results/figures/`. Analysis requires no API credentials or new model calls. The supplied plan is a redacted analytical derivative blocked from live collection. Collection/preparation code is retained for inspection and requires a new full configuration for a new experiment.

The filtering analysis is descriptive and post hoc. It reports complete/unanimous/non-unanimous counts, retention and disagreement rates. The training-majority baseline is also post hoc. These additions do not change the primary predictions or results. The supplied tests and numerical verification concern computational consistency, not independent authentication of provider responses.

## Source and scope

Reference labels and identifiers originate from [SemEval-2016 Task 6](https://www.saifmohammad.com/WebPages/StanceDataset.htm), described by [Mohammad et al. (2016)](https://doi.org/10.18653/v1/S16-1003). Obtain source post text from its creators and follow their terms. Hashes preserve record integrity and provenance. This historical benchmark is not a population survey; intervals are pointwise and there is no new independent human adjudication.

This repository is private. No license is assigned on behalf of the authors; third-party materials retain their own terms.
