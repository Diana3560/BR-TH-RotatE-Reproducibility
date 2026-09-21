# Reproducibility and limitations

## Reproducibility scope

The public repository preserves the model implementation, training/evaluation protocol, random-seed settings, statistical procedures, control experiments, and the numerical result tables used in the manuscript. Research datasets are deliberately maintained outside the code repository.

The principal CRH-L4MKG protocol uses a fixed 15,413 / 1,009 / 1,008 Train/Validation/Test split and ranks against 13,693 candidate entities under a both-side filtered evaluation. Main baseline results use seeds 42-46; the core D0/D1/A0/D2 ablation uses seeds 42-51. Repeated-split experiments use five relation-stratified repartitions with three training seeds per split.

## Data boundary

The repository does not include or modify the separately distributed anonymized data package. `data/` is documentation-only. Data-dependent runners require compatible files supplied independently by the researcher; the default public locations are under the ignored `external_data/` workspace.

## Interpretation limits

1. Random-seed comparisons quantify optimization variability conditional on a fixed dataset split.
2. Repeated splits are alternative partitions of the same historical graph, not independent external samples.
3. CROEFKG results require retraining on that dataset and do not represent zero-shot transfer.
4. MPNorm is a scale-control analysis rather than a performance-improvement method.
5. The frequency-shrinkage and coarse role-feature controls constrain the interpretation of the mechanism; neither is claimed as an independently proven universal source of performance gain.
6. Offline Hits@K/MRR do not directly measure field accuracy, maintenance acceptance rate, or operational safety.
7. The method is intended for candidate ranking and knowledge review, not for issuing maintenance instructions without evidence review and professional judgment.
