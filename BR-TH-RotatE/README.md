# BR-TH-RotatE

**Bounded Relation-adaptive TH-RotatE for railway-maintenance knowledge graph link prediction**

This repository contains the model implementation, experiment code, tests, and curated numerical outputs associated with the BR-TH-RotatE manuscript. It is a **code-and-results repository**: datasets are not bundled with the source tree and are not automatically connected to it.

## Overview

BR-TH-RotatE extends the TransH/RotatE hybrid scoring framework with a low-capacity relation-adaptive fusion mechanism. A shared global fusion state acts as an anchor, while Train-derived relation features generate a bounded relation-level angular offset. A frequency-dependent reliability factor further limits the available offset for low-frequency relations. Reciprocal training is used as a separate directional-supervision mechanism.

The four principal variants are:

| Variant | Definition |
| --- | --- |
| D0 | TH-RotatE baseline |
| D1 | D0 + reciprocal training |
| A0 | D0 + bounded relation adaptation |
| D2 | Reciprocal training + bounded relation adaptation (BR-TH-RotatE) |

For relation \(r\), the adaptive fusion is parameterized as

\[
\theta_r = \theta_g + q_r\Delta\tanh(g(z_r)),\qquad
\alpha_r=\cos\theta_r,\quad \beta_r=\sin\theta_r.
\]

The relation adapter is a shared bias-free linear map. On CRH-L4MKG, D2 adds 10 trainable parameters relative to D1.

## Repository scope

```text
BR-TH-RotatE/
├── config/                  # Experiment protocol and reproducibility metadata
├── data/                    # Dataset notice only; no dataset files
├── docs/                    # Method/reproducibility notes
├── results/manuscript/      # Curated manuscript-aligned numerical outputs
├── scripts/                 # Audits and experiment entry points
├── src/throtate_repro/      # Core model and training implementation
├── stage3/                  # Baseline and candidate-space control experiments
├── supplementary/           # Final supplementary/control experiment framework
├── tests/                   # Core tests
├── tests_stage3/            # Baseline/control tests
└── tests_supplementary/     # Supplementary experiment tests
```

Development caches, one-click launchers, executable files, local machine paths, raw/repeated-split dataset files, and internal delivery manifests are intentionally excluded from the public release.

## Experimental protocol

### CRH-L4MKG

- Complete graph: 24,768 triples, 17 relation types.
- R14 inference layer: 17,430 triples, 14 modeled relations.
- Fixed split: 15,413 / 1,009 / 1,008 (Train / Validation / Test).
- Full candidate set: 13,693 entities.
- Both-side filtered link prediction.
- Embedding dimension: 200 (TH-RotatE-Cap uses 201 for capacity control).
- Adam, learning rate 0.001.
- NSSA loss, margin 3, adversarial temperature 1.
- Batch size 1,024; 64 unfiltered Bernoulli negatives per positive.
- Exact budget: 3,000 optimizer updates; `drop_last=True`.
- Main baseline comparison: seeds 42-46.
- Core D0/D1/A0/D2 ablation: seeds 42-51.
- A0 maximum angle: 0.10 rad; D2 maximum angle: 0.15 rad.

### Additional validation

- CROEFKG external-data experiment: seeds 42-46; D2 angle bound 0.05 rad.
- Repeated-split robustness: 5 relation-stratified splits x 3 training seeds.
- MPNorm scale control: seeds 42-46.
- Frequency-shrinkage control: seeds 42-51.
- Equal-capacity role-feature control: seeds 42-46.
- Efficiency measurement: 5 randomized repeats per model, 20 warm-up + 200 measured updates.

## Manuscript-aligned results

The repository does not rely on historical raw run directories for the public numerical record. Final manuscript values are collected under `results/manuscript/`.

### Main CRH-L4MKG comparison (5 seeds)

| Model | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TransH | 0.525086 | 363.285 | 0.434524 | 0.573810 | 0.715476 |
| RotatE | 0.518704 | 1282.743 | 0.446925 | 0.552778 | 0.678472 |
| TH-RotatE (D0) | 0.552339 | 252.648 | 0.471925 | 0.592857 | 0.726488 |
| RatE | 0.532055 | 1186.332 | 0.454067 | 0.566964 | 0.713690 |
| CompoundE | 0.554453 | 632.358 | 0.474901 | 0.591567 | 0.729861 |
| PairRE | 0.539217 | 826.235 | 0.458730 | 0.578968 | 0.711508 |
| TH-RotatE-Cap | 0.551993 | 250.975 | 0.470040 | 0.595040 | 0.726786 |
| BR-TH-RotatE (D2) | 0.556818 | 191.206 | 0.474206 | 0.600298 | 0.734325 |

### Core ablation (10 seeds)

| Variant | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 0.552470 | 252.649 | 0.471726 | 0.592560 | 0.726240 |
| D1 | 0.554014 | 228.083 | 0.469395 | 0.599554 | 0.732887 |
| A0 | 0.556009 | 209.192 | 0.477778 | 0.593750 | 0.726042 |
| D2 | 0.556860 | 196.939 | 0.474504 | 0.600198 | 0.733532 |

For D2-D0, the paired 10-seed MRR difference is +0.004390 (95% CI 0.001743 to 0.007038; Holm-adjusted p=0.0136). The D2-D1 comparison remains a positive trend but does not reach the 0.05 threshold after correction.

See `docs/MANUSCRIPT_RESULT_MAP.md` for the exact file corresponding to each manuscript table and control analysis.

## Environment

All experiments reported in the manuscript were run in the following environment:

- Python 3.10.18
- PyTorch 2.9.0.dev20250810+cu128
- PyKEEN 1.11.1
- NumPy 1.26.4
- CUDA 12.8
- NVIDIA GeForce RTX 5070 Laptop GPU

The release uses one dependency specification for the main, Stage 3, and supplementary experiments. `requirements_stage3.txt` and `requirements_supplementary.txt` both reference `requirements.txt`, so all experiment entry points use the same package versions.

Install the dependencies with Python 3.10.18:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned PyTorch build is the CUDA 12.8 nightly build used to produce the manuscript results. A compatible NVIDIA driver and CUDA-capable GPU are required for CUDA execution.

## Data separation

No research dataset is included in this repository. In particular:

- `data/` contains only `DATA_README.md`;
- the separately prepared anonymized dataset package is **not embedded in, copied into, or modified by this repository**;
- no repository script automatically searches for, extracts, or reads that package by archive name;
- generated data or local authorized copies belong outside version control under the ignored `external_data/` workspace (or another path chosen by the researcher).

The experiment configuration retains expected counts and SHA-256 values as protocol metadata. Data-dependent runs fail or skip cleanly when external files are absent.

## Running experiments

The public repository can be inspected and tested without the datasets. Model training requires separately available compatible local data. The legacy command-line dataset keys are `ownkg` = CRH-L4MKG and `paper4` = CROEFKG; these identifiers do not imply that either dataset is bundled with the code.

In a fresh public clone, `python scripts/check_multidataset_data.py` reports that external data are absent; this is expected. Only after a researcher manually configures compatible local files should the data preflight and training workflow be run.

Main workflow after external data are configured:

```bash
python scripts/check_multidataset_data.py
python scripts/run_multidataset_experiment.py --dataset ownkg --stage preflight
python scripts/run_multidataset_experiment.py --dataset ownkg --stage screen
python scripts/run_multidataset_experiment.py --dataset ownkg --stage freeze
python scripts/run_multidataset_experiment.py --dataset ownkg --stage test
python scripts/run_multidataset_experiment.py --dataset ownkg --stage summarize
```

Baseline/candidate controls:

```bash
python stage3/run_stage3.py --dataset ownkg --stage all --device cuda
python stage3/run_candidate_sensitivity.py --dataset ownkg --stage all --device cuda
```

Final supplementary experiments:

```bash
python run_supplementary_experiments.py --stage core
python run_supplementary_experiments.py --stage splits
python run_supplementary_experiments.py --stage baselines
python run_supplementary_experiments.py --stage mpnorm
python run_supplementary_experiments.py --stage shrinkage
python run_supplementary_experiments.py --stage roles
python run_supplementary_experiments.py --stage efficiency
```

These commands do not download or alter the separately distributed dataset package.

## Verification

Run the release checks before creating a GitHub tag:

```bash
python scripts/check_manuscript_alignment.py
python scripts/verify_package.py
pytest -q
```

`check_manuscript_alignment.py` checks the final manuscript numerical anchors against `results/manuscript/`. `verify_package.py` checks release structure, code/config parseability, absence of bundled datasets and executables, and the release manifest.

## Reproducibility boundaries

Random-seed analyses quantify training variability under the stated splits and protocol. Repeated splits are repartitions of the same historical graph rather than independent external datasets. The model is evaluated for offline link prediction/candidate ranking and is not a substitute for maintenance rules or professional review. Additional limitations are summarized in `docs/REPRODUCIBILITY_AND_LIMITATIONS.md`.

## Citation

Please cite the associated BR-TH-RotatE manuscript when using this code. Bibliographic metadata can be added here after the manuscript receives its final publication information.
