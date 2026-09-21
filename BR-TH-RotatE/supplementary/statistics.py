from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from supplementary.protocol import METRICS, read_json, write_json, sha


def table(path, rows):
    if not rows: return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def summary(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else None}

def paired(a,b):
    x=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    s=summary(x); n=len(x)
    if n<2: ci=[None,None]; p=None;dz=None
    elif np.all(x==x[0]):
        ci=[float(x[0])]*2; p=1. if x[0]==0 else 0.;dz=None
    else:
        half=float(stats.t.ppf(.975,n-1)*stats.sem(x))
        ci=[s['mean']-half,s['mean']+half];p=float(stats.ttest_1samp(x,0).pvalue)
        dz=s['mean']/s['std']
    # Enumerated sign flips complement the paired t test for small samples.
    sign_p=None
    if 1<=n<=16:
        masks=np.arange(2**n,dtype=np.uint32)[:,None]
        signs=2*((masks>>np.arange(n))&1).astype(float)-1
        sign_p=float(np.mean(np.abs((signs*x).mean(axis=1))>=abs(x.mean())-1e-14))
    return {'n_pairs':n,'mean_delta':s['mean'],'std_delta':s['std'],'ci95_low':ci[0],'ci95_high':ci[1],
            'p_t_two_sided':p,'p_sign_flip_two_sided':sign_p,'cohen_dz':dz,
            'positive':int((x>0).sum()),'zero':int((x==0).sum()),'negative':int((x<0).sum())}

def holm(rows, key='p_t_two_sided', out='p_holm'):
    valid=[(i,float(r[key])) for i,r in enumerate(rows) if r.get(key) is not None]
    order=sorted(valid,key=lambda z:z[1]); running=0.
    for j,(i,p) in enumerate(order):
        running=max(running,min(1.,(len(order)-j)*p));rows[i][out]=running
    return rows

def bootstrap_differences(diffs,repeats,seed,unit):
    """diffs[seed, triple, direction]; keep all paired model observations aligned."""
    d=np.asarray(diffs,dtype=float)
    if d.ndim!=3 or d.shape[2]!=2: raise ValueError('Expected [seed,triple,head/tail]')
    rng=np.random.default_rng(seed); samples=[]
    if unit=='query': values=d.mean(axis=0).reshape(-1)
    elif unit=='triple': values=d.mean(axis=(0,2))
    elif unit!='seed_and_triple': raise ValueError(unit)
    for start in range(0,repeats,64):
        b=min(64,repeats-start)
        if unit=='seed_and_triple':
            seed_idx=rng.integers(d.shape[0],size=(b,d.shape[0]))
            # Resample training seeds and triple clusters independently (crossed design).
            v=d.mean(axis=2)[seed_idx].mean(axis=1)
            ids=rng.integers(d.shape[1],size=(b,d.shape[1]))
            samples.extend(np.take_along_axis(v,ids,axis=1).mean(axis=1).tolist())
        else:
            ids=rng.integers(len(values),size=(b,len(values)))
            samples.extend(values[ids].mean(axis=1).tolist())
    lo,hi=np.quantile(samples,[.025,.975])
    return {'unit':unit,'repeats':repeats,'mean_delta':float(d.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
            'interpretation':'Conditional sampling sensitivity; triples may remain graph-dependent; not fresh-data generalization.'}

def load_ranks(out,row):
    p=out/row['ranks_file']
    if sha(p)!=row['ranks_sha256']: raise RuntimeError(f'Rank file checksum mismatch: {p}')
    with np.load(p,allow_pickle=False) as f:return f['triples'].copy(),f['ranks'].copy()

def summarize_group(out,label,rows,comparisons,config):
    dest=out/'tables'/label; models=sorted({r['model'] for r in rows})
    raw=[{'model':r['model'],'seed':r['seed'],**r['metrics'],'train_seconds':r['train_seconds'],
          'parameters_real':r['parameter_counts']['trainable_real_scalar_parameters']} for r in rows]
    table(dest/'per_seed.csv',raw)
    aggregates=[]
    for m in models:
        row={'model':m}
        for metric in METRICS:
            s=summary([r[metric] for r in raw if r['model']==m]);row.update({metric+'_'+k:v for k,v in s.items()})
        aggregates.append(row)
    table(dest/'summary.csv',aggregates)
    lookup={(r['model'],r['seed']):r for r in rows}; tests=[]; boot=[]
    for a,b in comparisons:
        seeds=sorted(s for m,s in lookup if m==a and (b,s) in lookup)
        if not seeds:continue
        tests.append({'comparison':a+' minus '+b,'metric':'mrr',**paired(
            [lookup[a,s]['metrics']['mrr'] for s in seeds],[lookup[b,s]['metrics']['mrr'] for s in seeds])})
        diffs=[]; reference=None
        for s in seeds:
            qa,ra=load_ranks(out,lookup[a,s]);qb,rb=load_ranks(out,lookup[b,s])
            if not np.array_equal(qa,qb):raise ValueError('Unaligned paired queries')
            if reference is not None and not np.array_equal(reference,qa): raise ValueError('Across-seed query mismatch')
            reference=qa;diffs.append(1/ra-1/rb)
        for unit in ['query','triple','seed_and_triple']:
            boot.append({'comparison':a+' minus '+b,**bootstrap_differences(diffs,config['bootstrap_repeats'],config['bootstrap_seed'],unit)})
    holm(tests);holm(tests,'p_sign_flip_two_sided','p_sign_flip_holm')
    table(dest/'paired_seed_tests.csv',tests);table(dest/'paired_bootstrap.csv',boot)
    write_json(dest/'results.json',{'summary':aggregates,'paired':tests,'bootstrap':boot,
        'holm_family':label,'primary_metric':'MRR','secondary_metrics':'descriptive only; no significance claims from uncorrected secondary metrics'})
    return aggregates

def summarize_splits(out,rows):
    lookup={(r['split_seed'],r['model'],r['seed']):r['metrics'] for r in rows}
    per_split=[]
    for split in sorted({r['split_seed'] for r in rows}):
        for model in ['D0','D1','A0','D2']:
            values=[v for (sp,m,s),v in lookup.items() if sp==split and m==model]
            if not values:continue
            row={'split_seed':split,'model':model,'n_training_seeds':len(values)}
            for k in METRICS:
                ag=summary([v[k] for v in values]);row[k]=ag['mean'];row[k+'_within_split_std']=ag['std']
            per_split.append(row)
    table(out/'tables/repeated_splits/per_split.csv',per_split)
    table(out/'tables/repeated_splits/summary.csv',[
        {'model':m,**{k+'_'+field:value for k in METRICS for field,value in summary([r[k] for r in per_split if r['model']==m]).items()}}
        for m in sorted({r['model'] for r in per_split})])
    deltas=[]
    for a,b in [('A0','D0'),('D2','D1'),('D2','D0')]:
        x={r['split_seed']:r for r in per_split if r['model']==a}; y={r['split_seed']:r for r in per_split if r['model']==b}
        ids=sorted(x.keys()&y.keys())
        if ids:deltas.append({'comparison':a+' minus '+b,**paired([x[s]['mrr'] for s in ids],[y[s]['mrr'] for s in ids]),
             'inference_note':'DESCRIPTIVE sensitivity; overlapping splits are not independent experimental replicates.'})
    table(out/'tables/repeated_splits/descriptive_split_deltas.csv',deltas)

def stability(out,rows,train_counts):
    # Frequency tertiles from Train only, tied frequencies remain in the same bin.
    vals=np.array(list(train_counts.values()),dtype=float);q1,q2=np.quantile(vals,[1/3,2/3])
    groups={r:'low' if n<=q1 else 'middle' if n<=q2 else 'high' for r,n in train_counts.items()}
    table(out/'tables/shrinkage/frequency_groups.csv',[{'relation':r,'train_count':n,'group':groups[r],
          'low_threshold':q1,'middle_threshold':q2} for r,n in train_counts.items()])
    relation_rows=[]; grouped=[]; angles=[]
    for run in rows:
        for rr in run['per_relation']:
            relation_rows.append({'model':run['model'],'seed':run['seed'],'group':groups[rr['relation']],**rr})
        for group in ['low','middle','high']:
            values=[x for x in run['per_relation'] if groups[x['relation']]==group]
            if not values:continue
            grouped.append({'model':run['model'],'seed':run['seed'],'group':group,'n_relations':len(values),
                'micro_mrr':float(np.average([v['mrr'] for v in values],weights=[v['n_triples'] for v in values])),
                'macro_mrr':float(np.mean([v['mrr'] for v in values]))})
        for a in run['relation_fusion_state']:
            angles.append({'model':run['model'],'seed':run['seed'],'group':groups[a['relation']],**a})
    table(out/'tables/shrinkage/group_per_seed.csv',grouped)
    table(out/'tables/shrinkage/relation_per_seed.csv',relation_rows)
    table(out/'tables/shrinkage/angle_per_seed.csv',angles)
    variances=[]
    for model in sorted({r['model'] for r in rows}):
        for group in ['low','middle','high']:
            v=[r for r in grouped if r['model']==model and r['group']==group]
            if v:
                variances.append({'model':model,'group':group,'n_seeds':len(v),
                    **{k+'_'+f:x for k in ['micro_mrr','macro_mrr'] for f,x in summary([r[k] for r in v]).items()},
                    'micro_mrr_variance':float(np.var([r['micro_mrr'] for r in v],ddof=1)) if len(v)>1 else None})
    table(out/'tables/shrinkage/group_stability.csv',variances)
    av=[]
    for key in sorted({(r['model'],r['relation'],r['direction']) for r in angles}):
        a=[r for r in angles if (r['model'],r['relation'],r['direction'])==key]
        ag=summary([r['theta_shift_radians'] for r in a]);av.append({'model':key[0],'relation':key[1],
            'direction':key[2],'group':groups[key[1]],**ag,'variance':float(np.var([r['theta_shift_radians'] for r in a],ddof=1)) if len(a)>1 else None})
    table(out/'tables/shrinkage/angle_stability.csv',av)
    relation_var=[]
    for key in sorted({(r['model'],r['relation']) for r in relation_rows}):
        a=[r for r in relation_rows if (r['model'],r['relation'])==key]
        relation_var.append({'model':key[0],'relation':key[1],'group':groups[key[1]],
            **summary([r['mrr'] for r in a]),'variance':float(np.var([r['mrr'] for r in a],ddof=1)) if len(a)>1 else None})
    table(out/'tables/shrinkage/relation_stability.csv',relation_var)
    pairs=[]
    for group in ['low','middle','high']:
        a=next((r for r in variances if r['model']=='D2' and r['group']==group),None)
        b=next((r for r in variances if r['model']=='D2_noShrinkage' and r['group']==group),None)
        if a and b:
            va=a['micro_mrr_variance'];vb=b['micro_mrr_variance']
            pairs.append({'group':group,'variance_D2':va,'variance_noShrinkage':vb,
                'variance_ratio_D2_over_noShrinkage':va/vb if va is not None and vb else None,
                'interpretation':'Descriptive cross-seed stability; <1 favors shrinkage, not a significance test.'})
    table(out/'tables/shrinkage/variance_ratios.csv',pairs)
