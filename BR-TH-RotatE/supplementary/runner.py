from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from supplementary.protocol import CORE, read_json, write_json, freeze, source_hashes, digest, sha, prepare_split, settings, tuning_candidates, workload

STAGES=['core','splits','baselines','mpnorm','shrinkage','roles','efficiency']

def parse_args():
    p=argparse.ArgumentParser(description='Supplementary experiments; all formal runs use frozen protocol and full filtered ranking.')
    p.add_argument('--config',default='config/supplementary_experiments.json')
    p.add_argument('--stage',choices=['all','plan','summary']+STAGES,default='all')
    p.add_argument('--device',default='auto',help='auto, cpu, cuda, cuda:0')
    p.add_argument('--output',default=None)
    p.add_argument('--smoke',action='store_true',help='Synthetic tiny end-to-end checks; NEVER publication results')
    return p.parse_args()

def validate(config):
    for key in ['core_seeds','split_seeds','split_training_seeds','baseline_seeds','tuning_seeds','mechanism_seeds']:
        x=config[key]
        if not x or len(set(x))!=len(x) or any(not isinstance(v,int) for v in x):raise ValueError(f'Invalid seeds: {key}')
    if 42 not in config['core_seeds']:raise ValueError('Seed 42 required for fixed Train-only scale calibration')
    if len(config['tuning_seeds'])!=2 or config['shortlist_size']!=2:raise ValueError('Protocol specifies two tuning seeds and top two candidates')
    if not set(config['mechanism_seeds'])<=set(config['core_seeds']):raise ValueError('Mechanism seeds must be a subset of core seeds')
    if config['baseline_models']!=['RatE','RatE_recip','CompoundE','CompoundE_recip','PairRE','PairRE_recip','D2_tuned']:
        raise ValueError('Keep all seven baseline arms for the matched tuning comparison')
    for k in ['max_steps','batch_size','embedding_dim','num_negatives','evaluation_batch_size','evaluation_slice_size',
              'bootstrap_repeats','efficiency_repeats','efficiency_warmup_steps','efficiency_measured_steps','torch_num_threads']:
        if not isinstance(config[k],int) or config[k]<=0:raise ValueError(f'{k} must be a positive integer')
    if config['embedding_dim']%2:raise ValueError('CompoundE requires an even dimension')

def main():
    args=parse_args();root=Path(__file__).resolve().parents[1]
    config=read_json(root/args.config);validate(config)
    if args.smoke:
        config.update(embedding_dim=8,max_steps=3,batch_size=16,num_negatives=2,evaluation_batch_size=4,
            evaluation_slice_size=8,core_seeds=[42,43],split_seeds=[2026091401,2026091402],
            split_training_seeds=[42,43],baseline_seeds=[42,43],mechanism_seeds=[42,43],
            bootstrap_repeats=100,efficiency_repeats=2,efficiency_warmup_steps=1,efficiency_measured_steps=2)
    if args.stage=='plan':
        print(json.dumps({'mode':'SMOKE' if args.smoke else 'FORMAL','stages':STAGES,'config':config,
             'workload_upper_bound':workload(config),'steps_each_full_training':config['max_steps'],
             'note':'No training / no test scoring. Completed job checkpoints and ranks will be reused.'},ensure_ascii=False,indent=2));return
    import torch
    import yaml
    from importlib.metadata import version
    from throtate_repro.multidataset_experiment import _build_context
    from supplementary.engine import Engine,runtime
    from supplementary.statistics import summarize_group,summarize_splits,stability,table,summary
    if version('pykeen')!='1.11.1':raise RuntimeError('Require PyKEEN 1.11.1')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this Python environment')
    torch.set_num_threads(config['torch_num_threads'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if hasattr(torch.backends,'cuda'):torch.backends.cuda.matmul.allow_tf32=False
    # Fail rather than silently accepting a non-deterministic operation.
    torch.use_deterministic_algorithms(True)
    base=yaml.safe_load((root/'config/multidataset_comparison.yaml').read_text(encoding='utf-8'))
    base['training'].update({k:config[k] for k in ['max_steps','batch_size','num_negatives','evaluation_batch_size']})
    base['model']['embedding_dim']=config['embedding_dim'];base['training']['device']=device
    base['evaluation_slice_size_runtime']=config['evaluation_slice_size']
    fingerprints={'config':config,'source':source_hashes(root),'runtime':runtime(device),'smoke':args.smoke,
                  'original_data':base['datasets']['ownkg']['sha256']}
    identity=digest(fingerprints)
    out=(root/args.output).resolve() if args.output else root/'results'/('supplementary_SMOKE' if args.smoke else 'supplementary_runs')/identity[:16]
    out.mkdir(parents=True,exist_ok=True)
    # OS advisory lock: auto released after crash, no stale PID files / no parallel mutation.
    from filelock import FileLock,Timeout
    try: lock=FileLock(str(out/'RUN.lock'),timeout=0);lock.acquire()
    except Timeout:raise RuntimeError(f'Another runner is using {out}')
    try:
        freeze(out/'PROTOCOL_FROZEN.json',{'identity':identity,**fingerprints,
            'primary_seed_comparisons':['A0-D0','D2-D1','D2-D0'],
            'training_budget':'Exact optimizer steps, equal positive / negative exposure; no early stopping',
            'selection':'Baseline staged 8 candidates seed 1, top 2 repeated seed 2, mean Validation MRR',
            'fixed_core':'Paper gamma=3, A0 Delta=.10, D2 Delta=.15; no re-selection',
            'split_interpretation':'Repartitioning historical data cannot remove historical model-development exposure',
            'data_quality_scope':'This runner verifies dataset integrity and split protocol; semantic annotation quality is outside its scope.'})
        write_json(root/'results'/('SUPPLEMENTARY_SMOKE_LATEST.json' if args.smoke else 'SUPPLEMENTARY_LATEST.json'),{'output_directory':str(out.relative_to(root)) if out.is_relative_to(root) else str(out),'identity':identity})
        if args.smoke:
            from supplementary.smoke_data import prepare
            base=prepare(base,out)
        print(f'MODE={"SMOKE NOT PAPER RESULTS" if args.smoke else "FORMAL"} DEVICE={device}\nOUTPUT={out}',flush=True)
        print('Training workload upper bound: '+json.dumps(workload(config)),flush=True)
        engine=Engine(root,out,identity,device)
        ctx_val=_build_context(root,base,'ownkg',out/'context',include_test=False)
        write_json(out/'DATA_PROTOCOL_CHECK.json',ctx_val['audit'])
        index_path=out/'RUN_INDEX.json';index=read_json(index_path) if index_path.exists() else {}
        stages=STAGES if args.stage=='all' else [args.stage]
        def checkpoint(stage,rows):index[stage]=rows;write_json(index_path,index)
        def core_run(name,seed):
            # Also save Validation for the overlapping tuning seeds: no retraining when D2 is a tuning candidate.
            hp=settings(config,name)
            if name=='D2' and seed in config['tuning_seeds']:
                engine.run(base,ctx_val,name,hp,seed,'validation',keep=True)
            return engine.run(base,ctx_test,name,hp,seed,'test')
        ctx_test=_build_context(root,base,'ownkg',out/'context',include_test=True)
        for stage in stages:
            print(f'\nSTAGE {stage}',flush=True)
            if stage=='core':
                rows=[]
                for seed in config['core_seeds']:
                    for name in CORE:rows.append(core_run(name,seed));checkpoint(stage,rows)
            elif stage=='splits':
                rows=[]
                for sp in config['split_seeds']:
                    cfg=prepare_split(root,base,out,sp)
                    ctx=_build_context(root,cfg,'ownkg',out/'splits'/str(sp)/'context',include_test=True)
                    write_json(out/'splits'/str(sp)/'protocol_check.json',ctx['audit'])
                    for seed in config['split_training_seeds']:
                        for name in CORE:
                            rows.append({**engine.run(cfg,ctx,name,settings(config,name),seed,'test'),'split_seed':sp});checkpoint(stage,rows)
            elif stage=='baselines':
                selection_path=out/'BASELINE_SELECTION_FROZEN.json'
                selections={};screenrows=[];tune=config['tuning_seeds']
                if selection_path.exists():selections=read_json(selection_path)['models']
                else:
                    for label in config['baseline_models']:
                        name='D2' if label=='D2_tuned' else label
                        candidates=tuning_candidates(config,label);scores=[]
                        for i,hp in enumerate(candidates):
                            row=engine.run(base,ctx_val,name,hp,tune[0],'validation',keep=True)
                            scores.append((row['metrics']['mrr'],row['metrics']['hits_at_1'],-i,i))
                            screenrows.append({'model':label,'candidate':i,'seed':tune[0],**hp,**row['metrics']})
                        finalists=[x[3] for x in sorted(scores,reverse=True)[:config['shortlist_size']]]
                        means=[]
                        for i in finalists:
                            row=engine.run(base,ctx_val,name,candidates[i],tune[1],'validation',keep=True)
                            first=next(r for r in screenrows if r['model']==label and r['candidate']==i and r['seed']==tune[0])
                            screenrows.append({'model':label,'candidate':i,'seed':tune[1],**candidates[i],**row['metrics']})
                            means.append(((row['metrics']['mrr']+first['mrr'])/2,
                                          (row['metrics']['hits_at_1']+first['hits_at_1'])/2,-i,i))
                        best=sorted(means,reverse=True)[0][3]
                        selections[label]={'selected_hp':candidates[best],'selected_candidate':best,'finalists':finalists,
                                           'selection_uses':'Validation only','validation_mrr_mean':sorted(means,reverse=True)[0][0]}
                    # All choices frozen before any baseline Test evaluation.
                    table(out/'tables/baselines/validation_search.csv',screenrows)
                    freeze(selection_path,{'protocol_identity':identity,'models':selections})
                rows=[]
                for label in config['baseline_models']:
                    name='D2' if label=='D2_tuned' else label;hp=selections[label]['selected_hp']
                    for seed in config['baseline_seeds']:
                        rows.append({**engine.run(base,ctx_test,name,hp,seed,'test'),'model':label});checkpoint(stage,rows)
                # Remove unused screening states, retaining compact validation records and ranks.
                for label in config['baseline_models']:
                    name='D2' if label=='D2_tuned' else label
                    for hp in tuning_candidates(config,label):
                        for seed in tune:engine.prune(base,ctx_val,name,hp,seed)
            elif stage=='mpnorm':
                reference=core_run('D2',42)
                scales=reference['train_only_scales']
                freeze(out/'MPNORM_CALIBRATION_FROZEN.json',{'calibration_model':'D2','seed':42,
                      'hp':settings(config,'D2'),'constants':scales,'common_for':['D0_MPNorm','D1_MPNorm','D2_MPNorm'],
                      'note':'Same frozen Train-only factors for all arms; each final model initialized and trained from scratch.'})
                rows=[]
                for seed in config['mechanism_seeds']:
                    for name in ['D0','D1','D2']:
                        rows.append(core_run(name,seed))
                        rows.append(engine.run(base,ctx_test,name+'_MPNorm',settings(config,name),seed,'test',scales));checkpoint(stage,rows)
            elif stage=='shrinkage':
                rows=[]
                for seed in config['core_seeds']:
                    rows.append(core_run('D2',seed));rows.append(engine.run(base,ctx_test,'D2_noShrinkage',settings(config,'D2'),seed,'test'));checkpoint(stage,rows)
            elif stage=='roles':
                rows=[]
                for seed in config['mechanism_seeds']:
                    for name in ['D2_constRole','D2_constNoRole']:
                        rows.append(engine.run(base,ctx_test,name,settings(config,'D2'),seed,'test'));checkpoint(stage,rows)
            elif stage=='efficiency':
                from supplementary.efficiency import benchmark
                checkpoint(stage,benchmark(engine,base,ctx_val,config))
            elif stage!='summary':raise ValueError(stage)
            report(out,index,config,ctx_val['audit']['relation_counts']['train'],args.smoke)
        report(out,index,config,ctx_val['audit']['relation_counts']['train'],args.smoke)
        print(f'Finished requested stage. Read {out / "REPORT.md"}',flush=True)
    finally:lock.release()

def report(out,index,config,counts,smoke):
    from supplementary.statistics import summarize_group,summarize_splits,stability
    comparisons={'core':[('A0','D0'),('D2','D1'),('D2','D0')],
        'mpnorm':[('D2_MPNorm','D0_MPNorm'),('D2_MPNorm','D1_MPNorm'),('D0_MPNorm','D0'),('D1_MPNorm','D1'),('D2_MPNorm','D2')],
        'shrinkage':[('D2','D2_noShrinkage')],'roles':[('D2_constRole','D2_constNoRole')],
        'baselines':([('D2_tuned',m) for m in config['baseline_models'] if m!='D2_tuned'] + [(m+'_recip',m) for m in ['RatE','CompoundE','PairRE']])}
    expected={'core':4*len(config['core_seeds']),'splits':4*len(config['split_seeds'])*len(config['split_training_seeds']),
        'baselines':len(config['baseline_models'])*len(config['baseline_seeds']),'mpnorm':6*len(config['mechanism_seeds']),
        'shrinkage':2*len(config['core_seeds']),'roles':2*len(config['mechanism_seeds']),
        'efficiency':3*config['efficiency_repeats']}
    lines=['# Supplementary experiment report','',
           '**Synthetic smoke-test output; do not use as publication results.**' if smoke else
           'Results correspond to the frozen protocol. A positive mean difference does not by itself imply statistical significance.', '',
           '| Module | Completed | Planned | Status |','|---|---:|---:|---|']
    for stage,n in expected.items():
        rows=index.get(stage,[]);complete=len(rows)==n
        lines.append(f'| {stage} | {len(rows)} | {n} | {"complete" if complete else "incomplete"} |')
        if not complete:continue
        signature=digest(rows)
        marker=out/'tables'/stage/'SUMMARY_INPUT.json'
        if marker.exists() and read_json(marker).get('digest')==signature:continue
        if stage in comparisons:summarize_group(out,stage,rows,comparisons[stage],config)
        if stage=='splits':summarize_splits(out,rows)
        if stage=='shrinkage':stability(out,rows,counts)
        write_json(marker,{'digest':signature})
    lines+=['','## Output guide','',
        '- `tables/core`: ten-seed core ablation, paired tests, exact sign-flip tests, and Holm correction.',
        '- `tables/repeated_splits`: training-seed means within each split, followed by across-split summaries. These are sensitivity analyses, not independent external datasets.',
        '- `tables/baselines`: model-specific Validation-only tuning and reciprocal controls. `D2_tuned` remains separate from the fixed manuscript D2 setting.',
        '- `tables/mpnorm`: D0/D1/D2 controls using the same frozen Train-only calibration constants.',
        '- `tables/shrinkage`: frequency-shrinkage mechanism control and relation-frequency stability summaries.',
        '- `tables/roles`: equal-capacity role-feature control using a constant-one coordinate.',
        '- `tables/efficiency`: repeated post-warm-up timing, throughput, and memory measurements; these short measurements are not extrapolated to full-training wall time.',
        '', '## Interpretation limits','',
        '- Repeated splits are repartitions of the same historical graph and should not be interpreted as independent external datasets.',
        '- Each experiment module defines its own multiple-comparison family. The three core MRR comparisons are prespecified.',
        '- The runner verifies file integrity and split protocol when external data are supplied. Semantic annotation quality is not inferred from these numerical checks.',
        '- RatE/CompoundE follow the project scoring framework; PairRE uses PyKEEN 1.11.1. Results are comparisons under this repository protocol, not claims of reproducing each original paper\'s best reported configuration.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
