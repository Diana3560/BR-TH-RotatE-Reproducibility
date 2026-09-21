from __future__ import annotations
import gc
import os
import platform
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import NSSALoss
from pykeen.utils import set_random_seed
from throtate_repro.exact_step_v266 import build_exact_step_training_loop_class
from throtate_repro.v266_runtime import AuditedBernoulliNegativeSampler
from throtate_repro.gamma3_canonical_v251 import runtime_loss_audit
from throtate_repro.multidataset_experiment import _relation_fusion_rows
from stage3.baseline_runner import _count_params
from supplementary.models import spec_for
from supplementary.protocol import digest, sha, read_json, write_json

class QueryEvaluator(RankBasedEvaluator):
    """Use PyKEEN's filtered realistic ranks, retain explicit query identities."""
    def __init__(self):
        super().__init__(filtered=True)
        self.queries = {}
    def process_scores_(self, hrt_batch, target, scores, true_scores=None, dense_positive_mask=None):
        if not torch.isfinite(true_scores).all(): raise RuntimeError('Non-finite true scores')
        if torch.isinf(scores).any(): raise RuntimeError('Infinite candidate scores')
        super().process_scores_(hrt_batch=hrt_batch,target=target,scores=scores,
                                true_scores=true_scores,dense_positive_mask=dense_positive_mask)
        ranks = self.ranks[target, 'realistic'][-1]
        for triple, rank in zip(hrt_batch.detach().cpu().tolist(), ranks):
            self.queries[tuple(triple)+(target,)] = float(rank)

def metrics(ranks):
    x = np.asarray(ranks,dtype=float)
    if not x.size or not np.isfinite(x).all() or (x < 1).any(): raise ValueError('Invalid ranks')
    return {'mrr':float(np.mean(1/x)), 'mr':float(x.mean()),
            **{f'hits_at_{k}':float(np.mean(x<=k)) for k in [1,3,10]}}

def synchronize(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)

def runtime(device):
    d = torch.device(device)
    return {'python':platform.python_version(),'torch':torch.__version__, 'pykeen':version('pykeen'),
            'numpy':np.__version__,'device':str(d),'device_name':torch.cuda.get_device_name(d) if d.type=='cuda' else platform.processor() or 'CPU',
            'cuda':torch.version.cuda,'threads':torch.get_num_threads(),'platform':platform.platform()}

def build_model(cfg, context, name, hp, seed, device, scales):
    set_random_seed(seed)
    spec = spec_for(cfg,context,name,hp,scales)
    factory = context['reciprocal_training' if spec['reciprocal'] else 'training']
    model = spec['model'](triples_factory=factory,random_seed=seed,
            loss=NSSALoss(margin=hp['gamma'],adversarial_temperature=hp['adversarial_temperature']),
            **spec['model_kwargs']).to(device)
    return model, factory, spec

def train_model(cfg, context, name, hp, seed, device, scales=None, callbacks=None):
    model, factory, spec = build_model(cfg,context,name,hp,seed,device,scales)
    tc = cfg['training']; budget = context['budgets']['reciprocal' if spec['reciprocal'] else 'ordinary']
    optimizer = torch.optim.Adam(params=model.get_grad_params(),lr=hp['learning_rate'])
    loop = build_exact_step_training_loop_class()(model=model,triples_factory=factory,optimizer=optimizer,
        automatic_memory_optimization=False,negative_sampler=AuditedBernoulliNegativeSampler,
        negative_sampler_kwargs={'num_negs_per_pos':tc['num_negatives'],'filtered':False},
        exact_max_steps=tc['max_steps'],steps_per_full_epoch=budget['steps_per_full_epoch'],
        final_epoch_batches=budget['final_epoch_batches'],exact_num_epochs=budget['num_epochs'])
    AuditedBernoulliNegativeSampler.reset_audit()
    synchronize(device)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    losses = loop.train(triples_factory=factory,num_epochs=budget['num_epochs'],batch_size=tc['batch_size'],
            sub_batch_size=tc['batch_size'],drop_last=True,num_workers=0,
            use_tqdm=False,use_tqdm_batch=False,callbacks=callbacks)
    synchronize(device); seconds=time.perf_counter()-started
    exposure = AuditedBernoulliNegativeSampler.audit_snapshot(num_negatives_per_positive=tc['num_negatives'])
    steps = int(loop.v266_optimizer_steps)
    if steps != tc['max_steps']: raise RuntimeError(f'Update mismatch: {steps}')
    if exposure['actual_positive_instances'] != steps*tc['batch_size']: raise RuntimeError('Positive exposure mismatch')
    if exposure['requested_negatives'] != steps*tc['batch_size']*tc['num_negatives']: raise RuntimeError('Negative exposure mismatch')
    if not exposure['all_negatives_accepted']: raise RuntimeError('Unexpected filtered training negatives')
    if not np.isfinite(losses).all(): raise RuntimeError('Non-finite training loss')
    loss_check=runtime_loss_audit(model,hp['gamma'])
    if not loss_check['pass']: raise RuntimeError('Loss configuration mismatch')
    meta={'actual_optimizer_steps':steps,'exposure':exposure,'train_seconds':seconds,
          'parameter_counts':_count_params(model),'loss_audit':loss_check,'last_epoch_loss':float(losses[-1]),
          'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type=='cuda' else None,
          'gpu_peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)) if torch.device(device).type=='cuda' else None,
          'reciprocal':spec['reciprocal'],'implementation':spec['implementation'],'runtime':runtime(device)}
    if spec.get('control_design') is not None:
        meta['control_design']=spec['control_design']
    del loop,optimizer
    return model,meta

def train_scales(model, context, device, batch_size=512):
    # Exactly both internal orientations; relation IDs mapped once, explicitly.
    triples=context['training'].mapped_triples.clone()
    if model.use_inverse_triples:
        forward=triples.clone(); forward[:,1]*=2
        inverse=forward[:,[2,1,0]].clone(); inverse[:,1]+=1
        triples=torch.cat([forward,inverse])
    sums=np.zeros(2); n=0
    model.eval()
    with torch.inference_mode():
        for batch in triples.split(batch_size):
            comp=model.score_components_hrt(batch.to(device))
            for i,k in enumerate(['transh_distance','rotate_distance']): sums[i]+=comp[k].double().sum().item()
            n+=len(batch)
    if n==0 or not np.isfinite(sums).all() or (sums<=0).any(): raise RuntimeError('Invalid calibration')
    return {'c_h':float(sums[0]/n),'c_r':float(sums[1]/n),'n_training_orientations':n,
            'source':'Training positives only','train_sha256':context['audit']['sha256']['train'],
            'used_validation_or_test_scores':False}

def evaluate(model, context, split, cfg, device):
    ev=QueryEvaluator(); model.eval()
    triples=context[split].mapped_triples
    truths=[context['training'].mapped_triples,context.get('filter_validation',context['validation']).mapped_triples]
    if split=='test': truths.append(context['test'].mapped_triples)
    synchronize(device); start=time.perf_counter()
    result=ev.evaluate(model=model,mapped_triples=triples,additional_filter_triples=truths,
        batch_size=cfg['training']['evaluation_batch_size'],slice_size=cfg['evaluation_slice_size_runtime'],
        targets=('head','tail'),use_tqdm=False)
    synchronize(device)
    keys=[tuple(x) for x in triples.tolist()]
    ranks=np.array([[ev.queries[k+('head',)],ev.queries[k+('tail',)]] for k in keys],dtype=np.float64)
    primary=metrics(ranks)
    if not np.isclose(primary['mrr'],result.get_metric('both.realistic.inverse_harmonic_mean_rank'),atol=1e-7):
        raise RuntimeError('Query rank aggregation differs from PyKEEN')
    if len(ev.queries)!=2*len(triples): raise RuntimeError('Query identity mismatch')
    rows=[]
    for rid in sorted(set(triples[:,1].tolist())):
        selected=ranks[triples[:,1].numpy()==rid]
        rows.append({'relation':context['relation_id_to_label'][rid],'relation_id':rid,'n_triples':len(selected),
                     **metrics(selected),'head_mrr':metrics(selected[:,0])['mrr'],'tail_mrr':metrics(selected[:,1])['mrr']})
    return {'metrics':primary,'head':metrics(ranks[:,0]),'tail':metrics(ranks[:,1]),
            'per_relation':rows,'evaluation_seconds':time.perf_counter()-start,
            'n_queries':2*len(triples),'ranking':'filtered realistic both-side full entity candidates'},ranks,triples.numpy()

class Engine:
    def __init__(self, root, out, identity, device):
        self.root=root; self.out=out; self.identity=identity; self.device=device
    def paths(self, cfg, context, name, hp, seed, scales=None):
        description={'protocol':self.identity,'data':context['audit']['sha256'],'model':name,
            'hp':hp,'seed':seed,'scales':scales,'training':cfg['training'],
            'eval_slice':cfg['evaluation_slice_size_runtime']}
        key=digest(description); folder=self.out/'jobs'/key
        return folder,description
    def ensure(self,cfg,context,name,hp,seed,scales=None):
        folder,description=self.paths(cfg,context,name,hp,seed,scales)
        folder.mkdir(parents=True,exist_ok=True)
        meta_path=folder/'trained.json'; weights=folder/'model.pt'
        if meta_path.exists():
            meta=read_json(meta_path)
            if meta['identity']!=description: raise RuntimeError('Cached identity mismatch')
            if weights.exists():
                if sha(weights)!=meta['weights_sha256']: raise RuntimeError('Checkpoint checksum mismatch')
                return folder,meta,None
            # Finished jobs no longer need a model unless a new split evaluation is requested.
            return folder,meta,None
        print(f'TRAIN {name} seed={seed} gamma={hp["gamma"]} lr={hp["learning_rate"]}',flush=True)
        model,meta=train_model(cfg,context,name,hp,seed,self.device,scales)
        meta['identity']=description
        if name=='D2' and seed==42: meta['train_only_scales']=train_scales(model,context,self.device)
        meta['relation_fusion_state']=_relation_fusion_rows(model,context)
        tmp=weights.with_suffix('.tmp'); torch.save(model.state_dict(),tmp); os.replace(tmp,weights)
        meta['weights_sha256']=sha(weights); write_json(meta_path,meta)
        return folder,meta,model
    def run(self,cfg,context,name,hp,seed,split,scales=None,keep=False):
        folder,description=self.paths(cfg,context,name,hp,seed,scales)
        result_path=folder/(split+'.json'); rank_path=folder/(split+'_ranks.npz')
        if result_path.exists():
            row=read_json(result_path)
            if row['identity']!=description or not rank_path.exists() or sha(rank_path)!=row['ranks_sha256']:
                raise RuntimeError(f'Cached result mismatch: {result_path}')
            print(f'REUSE {name} seed={seed} {split}',flush=True)
            return row
        folder,meta,model=self.ensure(cfg,context,name,hp,seed,scales)
        if model is None:
            weights=folder/'model.pt'
            if not weights.exists(): raise RuntimeError(f'Checkpoint pruned before requested evaluation: {folder}')
            model,_,_=build_model(cfg,context,name,hp,seed,self.device,scales)
            model.load_state_dict(torch.load(weights,map_location=self.device,weights_only=True))
        try:
            evaluation,ranks,triples=evaluate(model,context,split,cfg,self.device)
            with (folder/(split+'_ranks.tmp')).open('wb') as f: np.savez_compressed(f,ranks=ranks,triples=triples)
            os.replace(folder/(split+'_ranks.tmp'),rank_path)
            row={**meta,**evaluation,'status':'PASS','model':name,'seed':seed,'hp':hp,
                 'split':split,'split_sha256':context['audit']['sha256'][split],
                 'ranks_file':str(rank_path.relative_to(self.out)),'ranks_sha256':sha(rank_path),
                 'test_used_for_parameter_selection':False}
            write_json(result_path,row)
            if not keep: (folder/'model.pt').unlink(missing_ok=True)
            print(f'DONE {name} seed={seed} {split} MRR={row["metrics"]["mrr"]:.6f}',flush=True)
            return row
        finally:
            del model;gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    def prune(self,cfg,context,name,hp,seed,scales=None):
        folder,_=self.paths(cfg,context,name,hp,seed,scales)
        (folder/'model.pt').unlink(missing_ok=True)
