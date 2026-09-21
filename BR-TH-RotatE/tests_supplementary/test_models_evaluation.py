import copy
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("pykeen")
import torch
import yaml
from supplementary.smoke_data import prepare
from supplementary.models import spec_for,PlainMP
from supplementary.engine import build_model,evaluate,train_model
from supplementary.protocol import read_json,settings
from throtate_repro.multidataset_experiment import _build_context

@pytest.fixture
def context(tmp_path):
    root=Path(__file__).resolve().parents[1]
    base=yaml.safe_load((root/'config/multidataset_comparison.yaml').read_text())
    cfg=prepare(base,tmp_path)
    cfg['training'].update(max_steps=3,batch_size=16,num_negatives=2,evaluation_batch_size=4)
    cfg['model']['embedding_dim']=8;cfg['evaluation_slice_size_runtime']=8
    ctx=_build_context(root,cfg,'ownkg',tmp_path/'context',include_test=True)
    hp={'embedding_dim':8,'gamma':3.,'delta':.15,'learning_rate':.001,'adversarial_temperature':1.}
    torch.set_num_threads(2)
    return cfg,ctx,hp

def test_plain_mpnorm_has_exact_original_parameter_count(context):
    cfg,ctx,hp=context
    plain,_,_=build_model(cfg,ctx,'D0',hp,42,'cpu',None)
    mp,_,_=build_model(cfg,ctx,'D0_MPNorm',hp,42,'cpu',{'c_h':1.,'c_r':1.})
    assert sum(p.numel() for p in plain.parameters() if p.requires_grad)==sum(p.numel() for p in mp.parameters() if p.requires_grad)
    assert not hasattr(mp.interaction,'relation_router')
    mp.load_state_dict(plain.state_dict(),strict=False)
    batch=ctx['test'].mapped_triples
    assert torch.allclose(plain.score_hrt(batch),mp.score_hrt(batch))

def test_constant_role_control_changes_only_role_information(context):
    cfg,ctx,hp=context
    before=ctx['reciprocal_bundle'].features.clone()
    a=spec_for(cfg,ctx,'D2_constRole',hp)['model_kwargs']['relation_feature_tensor']
    b=spec_for(cfg,ctx,'D2_constNoRole',hp)['model_kwargs']['relation_feature_tensor']
    assert a.shape==b.shape
    assert a.shape[1]==11
    assert torch.equal(a[:,2:],b[:,2:])
    assert (b[:,:2]==0).all() and (a[:,-1]==1).all()
    assert torch.equal(before,ctx['reciprocal_bundle'].features)

@pytest.mark.parametrize('name',['D0','D1','A0','D2','RatE_recip','CompoundE_recip','PairRE','PairRE_recip'])
def test_exact_budget_and_filtered_head_tail_reference(context,name):
    cfg,ctx,hp=context
    model,meta=train_model(cfg,ctx,name,hp,42,'cpu')
    assert meta['actual_optimizer_steps']==3
    assert meta['exposure']['actual_positive_instances']==48
    assert meta['exposure']['requested_negatives']==96
    result,ranks,triples=evaluate(model,ctx,'test',cfg,'cpu')
    # Independent brute-force scoring and filtering, including reciprocal head routing.
    truth=torch.cat([ctx[k].mapped_triples for k in ['training','validation','test']]).tolist()
    model.eval()
    with torch.inference_mode():
        expected=[]
        for h,r,t in triples:
            pair=[]
            for side in ['head','tail']:
                query=torch.tensor([[h,r,t]])
                scores=model.predict(hrt_batch=query,target=side).flatten().clone()
                target=int(h if side=='head' else t);true=scores[target].clone()
                for ah,ar,at in truth:
                    if ar==r and ((side=='head' and at==t) or (side=='tail' and ah==h)):
                        idx=ah if side=='head' else at
                        if idx!=target:scores[idx]=float('nan')
                optimistic=1+(scores>true).sum().item();pessimistic=(scores>=true).sum().item()
                pair.append((optimistic+pessimistic)/2)
            expected.append(pair)
    np.testing.assert_array_equal(ranks,expected)


def test_filtered_ties_are_realistic_not_optimistic(context):
    cfg,ctx,hp=context
    model,_,_=build_model(cfg,ctx,'PairRE',hp,42,'cpu',None)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    _,ranks,triples=evaluate(model,ctx,'test',cfg,'cpu')
    truth=set(tuple(x) for k in ['training','validation','test'] for x in ctx[k].mapped_triples.tolist())
    for i,(h,r,t) in enumerate(triples):
        for col,side in enumerate(['head','tail']):
            known=sum(ar==r and (at==t if side=='head' else ah==h) for ah,ar,at in truth)
            candidates=ctx['full'].num_entities-known+1
            assert ranks[i,col]==(candidates+1)/2

@pytest.mark.parametrize('name',['D0','D1','A0','D2'])
def test_new_trainer_matches_original_pipeline(context,name):
    from throtate_repro.multidataset_experiment import train_and_evaluate
    cfg,ctx,hp=context
    model,meta=train_model(cfg,ctx,name,hp,42,'cpu')
    new,_,_=evaluate(model,ctx,'test',cfg,'cpu')
    original=train_and_evaluate(cfg=cfg,context=ctx,identity={'fingerprint':'test'},model_name=name,
        seed=42,gamma=hp['gamma'],delta=hp['delta'],evaluation_split='test')
    for key,value in new['metrics'].items():assert value==pytest.approx(original['metrics'][key],rel=1e-6,abs=1e-7)
