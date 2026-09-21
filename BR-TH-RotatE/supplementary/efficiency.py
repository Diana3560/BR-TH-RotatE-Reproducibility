from __future__ import annotations
import copy
import gc
import random
import time
import numpy as np
import torch
from pykeen.training.callbacks import TrainingCallback
from throtate_repro.v266_protocol import exact_step_budget
from supplementary.engine import train_model, evaluate, synchronize
from supplementary.protocol import read_json, write_json, settings
from supplementary.statistics import table, summary

class Clock(TrainingCallback):
    def __init__(self,warmup,measured,device):
        super().__init__();self.warmup=warmup;self.device=device;self.steps=0;self.start=None;self.elapsed=None;self.total=warmup+measured
    def post_batch(self,epoch,batch,**kwargs):
        self.steps+=1
        if self.steps==self.warmup:
            synchronize(self.device)
            if torch.device(self.device).type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
            self.start=time.perf_counter()
        if self.steps==self.total:
            synchronize(self.device);self.elapsed=time.perf_counter()-self.start
    def finish(self):
        synchronize(self.device)
        if self.start is None:raise RuntimeError('Warmup did not finish')
        if self.elapsed is None:raise RuntimeError("Measured updates did not finish")

def benchmark(engine,cfg,context,config):
    rows=[];directory=engine.out/'efficiency';directory.mkdir(exist_ok=True)
    for repeat in range(config['efficiency_repeats']):
        models=['D0','D1','D2'];random.Random(20260914+repeat).shuffle(models)
        for order,name in enumerate(models):
            p=directory/f'{name}_repeat{repeat}.json'
            if p.exists(): rows.append(read_json(p));continue
            steps=config['efficiency_warmup_steps']+config['efficiency_measured_steps']
            bc=copy.deepcopy(cfg);bc['training']['max_steps']=steps
            ctx=dict(context);ctx['budgets']={key:exact_step_budget(num_real_triples=context['training'].num_triples,
                batch_size=bc['training']['batch_size'],max_steps=steps,reciprocal=recip,drop_last=True)
                for key,recip in [('ordinary',False),('reciprocal',True)]}
            clock=Clock(config['efficiency_warmup_steps'],config['efficiency_measured_steps'],engine.device)
            print(f'BENCHMARK {name} repeat={repeat+1}/{config["efficiency_repeats"]}',flush=True)
            model,meta=train_model(bc,ctx,name,settings(config,name),42,engine.device,callbacks=[clock])
            clock.finish()
            # Small fixed Validation prefix, full candidate set, same for every model.
            v=context['validation'];small=v.clone_and_exchange_triples(v.mapped_triples[:min(32,v.num_triples)])
            evctx=dict(context);evctx['validation']=small;evctx['filter_validation']=v
            # Original complete validation remains in filtering to keep definitions equal.
            warm,_,_=evaluate(model,evctx,'validation',bc,engine.device)
            ev,_,_=evaluate(model,evctx,'validation',bc,engine.device)
            measured=config['efficiency_measured_steps']
            row={'model':name,'repeat':repeat,'execution_order':order,'seed':42,
                 'gamma':3.,'delta':.15 if name=='D2' else None,
                 'warmup_steps':config['efficiency_warmup_steps'],'measured_steps':measured,
                 'measured_train_seconds':clock.elapsed,'ms_per_update':clock.elapsed*1000/measured,
                 'positive_instances_per_second':measured*bc['training']['batch_size']/clock.elapsed,
                 'eval_seconds':ev['evaluation_seconds'],'eval_queries':ev['n_queries'],
                 'eval_queries_per_second':ev['n_queries']/ev['evaluation_seconds'],
                 **meta}
            write_json(p,row);rows.append(row)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    table(engine.out/'tables/efficiency/per_repeat.csv',rows)
    fields=['ms_per_update','positive_instances_per_second','eval_queries_per_second','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes']
    ag=[]
    for name in ['D0','D1','D2']:
        row={'model':name,'repeat_count':config['efficiency_repeats']}
        for k in fields:
            x=[r[k] for r in rows if r['model']==name and r[k] is not None]
            if x:row.update({k+'_'+a:b for a,b in summary(x).items()})
        ag.append(row)
    table(engine.out/'tables/efficiency/summary.csv',ag)
    return rows
