from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]
(root/"results/figures").mkdir(exist_ok=True)
import os
os.environ['MPLCONFIGDIR']='/tmp/stance-mpl'
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.titlesize':12,'axes.labelsize':11.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def rows(name):return list(csv.DictReader((root/'results'/name).open()))
models=['gpt4o_mini','qwen3_30b'];names=['GPT-4o mini','Qwen3 30B'];cond={'zero_shot':'Zero-shot','persona_neutral':'Neutral','persona_pro':'Pro','persona_con':'Con'}
short={'Atheism':'Atheism','Climate Change is a Real Concern':'Climate change','Feminist Movement':'Feminist movement','Hillary Clinton':'Hillary Clinton','Legalization of Abortion':'Abortion legalization'}
for num in [1,2]:
 fig,axes=plt.subplots(1,2,figsize=(6.85,3.2),layout='constrained')
 data=rows('consistency.csv') if num==1 else [x for x in rows('prevalence_shifts.csv') if x['left']=='persona_pro' and x['right']=='persona_con' and x['label']=='FAVOR']
 for j,(ax,model,name) in enumerate(zip(axes,models,names)):
  rr=[r for r in data if r['model']==model]
  keys=['unanimous_wrong_rate','unanimous_wrong_ci_low','unanimous_wrong_ci_high'] if num==1 else ['share_difference','ci_low','ci_high']
  for i,r in enumerate(rr):
   v,lo,hi=[100*float(r[k]) for k in keys];ax.plot([lo,hi],[i,i],color='#24557a',lw=1.2);ax.plot(v,i,'o',color='#24557a',ms=4)
  labels=[cond[r['condition']]+' (n='+r['n_unanimous']+')' for r in rr] if num==1 else [short[r['target']] for r in rr]
  ax.set_yticks(range(len(rr)),labels);ax.invert_yaxis();ax.set_ylim(len(rr)-.5,-.5);ax.set_title(f'({chr(97+j)}) {name}',loc='left');ax.set_xlim(25 if num==1 else 0,50 if num==1 else 30);ax.set_xticks([25,30,35,40,45,50] if num==1 else [0,10,20,30]);ax.set_xlabel('Disagreement (%)' if num==1 else 'FAVOR shift (pp)');ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
 for ext in ['png','pdf','eps','svg']:fig.savefig(root/'results/figures'/f'Fig{num}.{ext}',dpi=600,facecolor='white',metadata={'Creator':'','Title':f'Figure {num}'} if ext in ['pdf','eps'] else None)
 plt.close(fig)
