from pathlib import Path
import csv, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def triples(name):
    p=ROOT/'anonymized_data'/name
    with p.open(encoding='utf-8',newline='') as f:
        return [tuple(r) for r in csv.reader(f,delimiter='\t') if r]

def main():
    expected={}
    for line in (ROOT/'integrity/SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        h,rel=line.split('  ',1); expected[rel]=h
    bad=[]
    for rel,h in expected.items():
        p=ROOT/rel
        if not p.exists() or sha(p)!=h: bad.append(rel)
    tr,va,te=triples('R14_train.tsv'),triples('R14_valid.tsv'),triples('R14_test.tsv')
    sets=list(map(set,(tr,va,te)))
    rels=[set(x[1] for x in z) for z in (tr,va,te)]
    cand=[]
    with (ROOT/'anonymized_data/candidate_entities.tsv').open(encoding='utf-8') as f:
        rd=csv.reader(f,delimiter='\t'); next(rd); cand=[r[0] for r in rd]
    checks={
      'hashes_pass':not bad,
      'bad_hash_files':bad,
      'counts':[len(tr),len(va),len(te)],
      'duplicates':[len(tr)-len(sets[0]),len(va)-len(sets[1]),len(te)-len(sets[2])],
      'overlaps':[len(sets[0]&sets[1]),len(sets[0]&sets[2]),len(sets[1]&sets[2])],
      'all_splits_have_14_relations':all(len(x)==14 for x in rels),
      'candidate_entities':len(cand),
    }
    checks['status']='PASS' if checks['hashes_pass'] and checks['counts']==[15413,1009,1008] and checks['duplicates']==[0,0,0] and checks['overlaps']==[0,0,0] and checks['all_splits_have_14_relations'] and checks['candidate_entities']==13693 else 'FAIL'
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    sys.exit(0 if checks['status']=='PASS' else 1)
if __name__=='__main__': main()
