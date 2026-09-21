from __future__ import annotations
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

CORE = ['D0', 'D1', 'A0', 'D2']
METRICS = ['mrr', 'mr', 'hits_at_1', 'hits_at_3', 'hits_at_10']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def freeze(path, value):
    if Path(path).exists():
        if read_json(path) != value: raise RuntimeError(f'Frozen protocol mismatch: {path}; use a new output directory.')
    else: write_json(path, value)

def source_hashes(root):
    return {str(p.relative_to(root)).replace('\\', '/'): sha(p)
            for folder in ['src', 'stage3', 'supplementary'] for p in sorted((root / folder).rglob('*.py'))}

def read_triples(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        bits = line.split('\t')
        if len(bits) != 3: raise ValueError(f'Expected three TSV fields: {path}')
        rows.append(tuple(bits))
    return rows

def coverage_split(rows, seed, valid_fraction=.06, test_fraction=.06):
    """Relation-stratified holdout; remove only edges leaving both endpoints in Train.

    No label repair based on scores. Constrained holdout may be below requested size;
    report achieved fractions and per-relation deficits instead of hiding the constraint.
    """
    rows = sorted(rows)
    if len(set(rows)) != len(rows): raise ValueError('Duplicate input triples')
    if not (0 < valid_fraction < 1 and 0 < test_fraction < 1 and valid_fraction+test_fraction < 1):
        raise ValueError('Invalid split fractions')
    rng = random.Random(seed)
    counts = Counter(e for h, _, t in rows for e in {h, t})
    rc = Counter(r for _, r, _ in rows)
    by = defaultdict(list)
    for row in rows: by[row[1]].append(row)
    held = {'validation': [], 'test': []}; removed = set(); audit = {}
    relations = sorted(by); rng.shuffle(relations)
    for r in relations:
        candidates = by[r][:]; rng.shuffle(candidates)
        target = {'validation': max(1, round(len(candidates)*valid_fraction)),
                  'test': max(1, round(len(candidates)*test_fraction))}
        got = Counter()
        # Alternate subsets with randomized tie breaking, avoiding systematic side preference.
        for row in candidates:
            available = [s for s in held if got[s] < target[s]]
            if not available: break
            endpoints = {row[0], row[2]}
            if rc[r] <= 1 or any(counts[e] <= 1 for e in endpoints): continue
            rng.shuffle(available)
            s = min(available, key=lambda s: got[s]/target[s])
            held[s].append(row); got[s] += 1; removed.add(row); rc[r] -= 1
            for e in endpoints: counts[e] -= 1
        audit[r] = {'total': len(candidates), 'train': rc[r], 'requested': target,
                    'achieved': {s: got[s] for s in held}}
        if any(got[s] == 0 for s in held):
            raise ValueError(f'Relation {r} cannot populate both holdouts with Train endpoint coverage')
    result = {'train': [r for r in rows if r not in removed],
              **{k: sorted(v) for k, v in held.items()}}
    audit['fractions'] = {k: len(v)/len(rows) for k, v in result.items()}
    return result, audit

def prepare_split(root, base, out, split_seed):
    cfg = copy.deepcopy(base)
    if split_seed is None: return cfg
    spec = cfg['datasets']['ownkg']; old = root / spec['split_dir']
    all_rows = sum((read_triples(old / spec[k]) for k in ['train_filename','validation_filename','test_filename']), [])
    rows, details = coverage_split(all_rows, split_seed)
    dest = out / 'splits' / str(split_seed); dest.mkdir(parents=True, exist_ok=True)
    for name, filename in [('train','train.tsv'),('validation','valid.tsv'),('test','test.tsv')]:
        payload = ''.join('\t'.join(row)+'\n' for row in rows[name])
        p = dest / filename
        if p.exists() and p.read_text(encoding='utf-8') != payload: raise RuntimeError(f'Split drift: {p}')
        p.write_text(payload, encoding='utf-8')
        spec['sha256'][name] = sha(p)
        spec['expected'][{'train':'train_triples','validation':'validation_triples','test':'test_triples'}[name]] = len(rows[name])
        if name != 'train': spec['expected'][name+'_relations'] = len({r for _,r,_ in rows[name]})
    spec['split_dir'] = str(dest.resolve())
    write_json(dest/'split_manifest.json', {'seed':split_seed,'details':details,'sha256':spec['sha256'],
        'interpretation':'Repeated split robustness on previously available data; not fresh independent data.'})
    return cfg

def settings(config, model):
    return {'gamma':3.,'learning_rate':.001,'adversarial_temperature':1.,
            'embedding_dim':config['embedding_dim'],
            'delta': .10 if model == 'A0' else .15}

def tuning_candidates(config, model):
    # Eight candidates each; three tuning axes, fixed dimension / negative budget.
    candidates = []
    gammas = [3., 6., 12., 24.]
    for i, gamma in enumerate(gammas):
        for variant in [0, 1]:
            p = settings(config, model)
            p.update(gamma=gamma, learning_rate=.001 if variant == 0 else .0005,
                     adversarial_temperature=1. if variant == 0 else .5)
            if model.startswith('D2'): p['delta'] = [.05,.10,.15,.15][i] if variant else .15
            candidates.append(p)
    return candidates

def workload(config):
    b = len(config['baseline_models']); formal = len(config['baseline_seeds'])
    screen = 8 + config['shortlist_size'] * (len(config['tuning_seeds'])-1)
    return {'fixed_core':4*len(config['core_seeds']),
            'repeated_splits':4*len(config['split_seeds'])*len(config['split_training_seeds']),
            'baselines_screen':b*screen,
            'baselines_formal_max_additional':b*max(0, formal-len(set(config['baseline_seeds'])&set(config['tuning_seeds']))),
            'mpnorm':3*len(config['mechanism_seeds']),
            'shrinkage':len(config['core_seeds']),
            'role_controls':2*len(config['mechanism_seeds']),
            'efficiency_short_runs':3*config['efficiency_repeats']}
