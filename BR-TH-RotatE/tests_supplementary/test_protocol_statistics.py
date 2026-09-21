import numpy as np
import pytest
from supplementary.protocol import coverage_split,freeze,workload,read_json
from supplementary.statistics import paired,holm,bootstrap_differences

def test_split_preserves_coverage_and_is_reproducible():
    rows=[(str(i),r,str((i+k)%30)) for r in ['r1','r2','r3'] for k in [1,2,3] for i in range(30)]
    a,_=coverage_split(rows,123);b,_=coverage_split(list(reversed(rows)),123)
    assert a==b
    parts=[set(v) for v in a.values()]
    assert set.union(*parts)==set(rows)
    assert not any(parts[i]&parts[j] for i in range(3) for j in range(i))
    train_entities={e for h,r,t in a['train'] for e in [h,t]}
    for subset in ['validation','test']:
        assert {e for h,r,t in a[subset] for e in [h,t]}<=train_entities
        assert {r for h,r,t in a[subset]}=={'r1','r2','r3'}
    assert coverage_split(rows,124)[0]!=a

def test_impossible_coverage_is_not_silently_dropped():
    with pytest.raises(ValueError,match='cannot populate'):
        coverage_split([('one','rare','two')],1)

def test_freeze_blocks_configuration_drift(tmp_path):
    p=tmp_path/'protocol.json';freeze(p,{'delta':.15});freeze(p,{'delta':.15})
    with pytest.raises(RuntimeError,match='mismatch'):freeze(p,{'delta':.10})

def test_seed_statistics_and_holm_known_values():
    result=paired([.3,.5,.8],[.2,.3,.5]);assert result['mean_delta']==pytest.approx(.2)
    assert result['positive']==3
    assert result['p_sign_flip_two_sided']==.25
    x=holm([{'p_t_two_sided':.01},{'p_t_two_sided':.04},{'p_t_two_sided':.03}])
    assert [r['p_holm'] for r in x]==pytest.approx([.03,.06,.06])

def test_bootstrap_retains_triple_head_tail_pairing():
    # Opposite effects cancel exactly at the triple level, not at query level.
    d=np.tile([.1,-.1],(3,20,1))
    triple=bootstrap_differences(d,500,42,'triple')
    crossed=bootstrap_differences(d,500,42,'seed_and_triple')
    query=bootstrap_differences(d,500,42,'query')
    assert triple['ci95_low']==triple['ci95_high']==0
    assert crossed['ci95_low']==crossed['ci95_high']==0
    assert query['ci95_low']<0<query['ci95_high']

def test_bootstrap_identical_models_have_zero_delta():
    for mode in ['query','triple','seed_and_triple']:
        r=bootstrap_differences(np.zeros((2,7,2)),100,4,mode)
        assert r['ci95_low']==r['ci95_high']==r['mean_delta']==0
