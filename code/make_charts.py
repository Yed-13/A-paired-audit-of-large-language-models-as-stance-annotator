#!/usr/bin/env python3
"""Render source-data charts or verified live-result tables. Never plots mock results."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SHORT = {'Atheism':'Atheism','Climate Change is a Real Concern':'Climate change',
         'Feminist Movement':'Feminism','Hillary Clinton':'Hillary Clinton',
         'Legalization of Abortion':'Abortion legalization'}
COLORS = ['#0072B2','#D55E00','#999999']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42})


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(out / (name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def dataset(folder,out):
    status=json.loads((folder/'STATUS.json').read_text())
    if status['status']!='REAL_SOURCE_DATA_DESCRIPTIVE_ONLY':
        raise ValueError('Not a real source-data summary')
    rows=read(folder/'composition.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,split,title in zip(axes,['test_pool','main_sample'],['Full test pool (n = 1,249)','Selected study sample (n = 600)']):
        base=np.zeros(5)
        for label,color,hatch in zip(['FAVOR','AGAINST','NONE'],COLORS,['','///','...']):
            vals=np.array([sum(int(r['n']) for r in rows if r['split']==split and r['target']==t and r['label']==label) for t in SHORT])
            ax.barh(list(SHORT.values()),vals,left=base,color=color,label=label,hatch=hatch,edgecolor='white')
            for i,v in enumerate(vals):
                if v>=15: ax.text(base[i]+v/2,i,str(v),ha='center',va='center',color='white',fontsize=9,bbox={'facecolor':color,'edgecolor':'none','pad':1.5})
            base+=vals
        ax.set_title(title);ax.set_xlabel('Number of posts');ax.invert_yaxis()
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8,frameon=False)
    save(fig,out,'dataset_stance_composition')
    sample=json.loads((ROOT/'data/private/sample-600.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    base=np.zeros(5)
    for label,color,hatch in zip(['TARGET','OTHER','NO ONE'],COLORS,['','///','...']):
        vals=np.array([sum(r['target']==t and r['opinion_towards']==label for r in sample) for t in SHORT])
        ax.barh(list(SHORT.values()),vals,left=base,color=color,label=label,hatch=hatch,edgecolor='white')
        base+=vals
    ax.invert_yaxis();ax.set_xlabel('Number of selected posts')
    ax.set_title('Original human opinion-target annotations (n = 600)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8,frameon=False)
    save(fig,out,'dataset_opinion_target')
    (out/'captions.md').write_text('''# Dataset figures

**Dataset stance composition.** Original reference stance labels in the five-target test pool and the seeded 600-item sample. Each selected target contributes 120 posts. These are source-data descriptions, not model-performance results. NONE means neither directional stance can be inferred.

**Opinion target.** Original human opinion-target annotations for the selected posts. TARGET, OTHER and NO ONE describe the object of opinion; they are not an independently validated scale of inference difficulty. Source: official SemEval stance archive, with hash in the data provenance file.
''')


def results(folder,out):
    status=json.loads((folder/'STATUS.json').read_text())
    if status['status']!='OBSERVED_LIVE_RECORDS_NOT_EXTERNALLY_AUTHENTICATED':
        raise ValueError('Live-result charts refuse synthetic, pilot, or unknown analyses')
    perf=read(folder/'performance_by_run.csv'); consistency=read(folder/'consistency.csv')
    models=sorted({r['model'] for r in perf}); conditions=list(dict.fromkeys(r['condition'] for r in perf))
    fig,axes=plt.subplots(len(models),2,figsize=(11,4*len(models)),squeeze=False,layout='constrained')
    for row,model in enumerate(models):
        for ax,metric,title in zip(axes[row],['accuracy_all','coverage'],['Accuracy over all attempts','Valid response coverage']):
            for i,c in enumerate(conditions):
                v=[float(r[metric]) for r in perf if r['model']==model and r['condition']==c]
                ax.scatter(v,[i]*len(v),color=COLORS[0],alpha=.5)
                ax.scatter([np.mean(v)],[i],marker='D',color='black',s=22)
            ax.set_yticks(range(len(conditions)),conditions);ax.set_xlim(0,1.02)
            ax.set_title(model+' — '+title);ax.invert_yaxis()
    save(fig,out,'performance_and_coverage')
    fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4.5),squeeze=False,layout='constrained')
    for ax,model in zip(axes[0],models):
        selected=[r for r in consistency if r['model']==model]
        for i,r in enumerate(selected):
            y=float(r['unanimous_wrong_rate'])
            lo=float(r['unanimous_wrong_ci_low']);hi=float(r['unanimous_wrong_ci_high'])
            if np.isfinite(y):
                ax.scatter([y],[i],color=COLORS[1])
                if np.isfinite(lo) and np.isfinite(hi): ax.hlines(i,lo,hi,color=COLORS[1])
            else: ax.text(.01,i,'Undefined',va='center')
        ax.set_yticks(range(len(selected)),[r['condition']+' (n='+r['n_unanimous']+')' for r in selected])
        ax.set_xlim(0,1);ax.invert_yaxis();ax.set_title(model)
        ax.set_xlabel('Error rate among unanimous items (95% pointwise CI)')
    save(fig,out,'unanimous_errors')
    shifts=[r for r in read(folder/'prevalence_shifts.csv') if r['left']=='persona_pro' and r['right']=='persona_con' and r['label']=='FAVOR']
    fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4.5),squeeze=False,layout='constrained')
    for ax,model in zip(axes[0],models):
        selected=[r for r in shifts if r['model']==model]
        for i,r in enumerate(selected):
            mid,lo,hi=[float(r[k]) for k in ['share_difference','ci_low','ci_high']]
            ax.scatter([mid],[i],color=COLORS[0]);ax.hlines(i,lo,hi,color=COLORS[0])
        ax.set_yticks(range(len(selected)),[SHORT[r['target']] for r in selected]);ax.axvline(0,color='gray',ls='--')
        ax.set_xlim(-1,1);ax.invert_yaxis();ax.set_title(model)
        ax.set_xlabel('FAVOR share: pro minus con (95% pointwise CI)')
    save(fig,out,'persona_favor_shift')
    (out/'captions.md').write_text('''# Model-result figures

**Performance and coverage.** Points show individual rounds; black diamonds show round means. These marks are not confidence intervals. Failed attempts count as incorrect in all-attempt accuracy.

**Unanimous errors.** Error proportions among complete unanimous sequences, with target-stratified item-bootstrap pointwise 95% intervals. Labels give unanimous denominators. Undefined estimates remain undefined.

**Persona FAVOR shift.** Within-model pro-minus-con differences in FAVOR share on complete paired sequences, with pointwise 95% item-bootstrap intervals. These intervals are not multiplicity-adjusted. Other labels and contrasts remain available in the tables.
''')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['dataset','results'])
    ap.add_argument('--tables',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    (dataset if args.kind=='dataset' else results)(args.tables,args.out)
