from __future__ import annotations
import copy
from supplementary.protocol import coverage_split,sha

def prepare(base,out):
    cfg=copy.deepcopy(base);dest=out/'toy_data';dest.mkdir(parents=True,exist_ok=True)
    relations=['HAS_FAILURE','HAS_PART','NEXT_STEP']
    rows=[(f'e{i:02d}',r,f'e{(i+shift)%24:02d}') for j,r in enumerate(relations)
          for shift in [j+1,j+5,j+9] for i in range(24)]
    splits,_=coverage_split(rows,20260914)
    paths={'full':dest/'full.tsv','train':dest/'train.tsv','validation':dest/'valid.tsv','test':dest/'test.tsv','entity_types':dest/'types.tsv'}
    for k in ['full','train','validation','test']:
        paths[k].write_text(''.join('\t'.join(r)+'\n' for r in (rows if k=='full' else splits[k])),encoding='utf-8')
    paths['entity_types'].write_text('entity_id\tentity_type\n'+''.join(f'e{i:02d}\tEntity\n' for i in range(24)),encoding='utf-8')
    cfg['datasets']['ownkg']={'display_name':'SYNTHETIC SMOKE ONLY','identity_note':'Not a manuscript result',
        'full_file':str(paths['full']),'entity_types_file':str(paths['entity_types']),'split_dir':str(dest),
        'train_filename':'train.tsv','validation_filename':'valid.tsv','test_filename':'test.tsv',
        'modeled_relations':relations,'relation_roles':{r:'diagnostic_semantic' if i==0 else 'procedural_structural' for i,r in enumerate(relations)},
        'expected':{'candidate_entities':24,'mapping_relations':3,'modeled_relations':3,'full_triples':len(rows),
                    'train_triples':len(splits['train']),'validation_triples':len(splits['validation']),'test_triples':len(splits['test']),
                    'split_union_triples':len(rows),'validation_relations':3,'test_relations':3},
        'sha256':{k:sha(p) for k,p in paths.items()}}
    return cfg
