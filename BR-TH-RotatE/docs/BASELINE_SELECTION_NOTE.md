# Baseline selection and tuning note

The baseline set covers complementary KGE families while keeping the comparison protocol controlled: TransH, RotatE, TH-RotatE (D0), RatE, CompoundE, PairRE, and the capacity-matched TH-RotatE-Cap control. Reciprocal variants of RatE, CompoundE, and PairRE are included in the supplementary tuning framework as fairness controls.

All reported baseline comparisons use the project's shared optimizer/update budget and evaluation protocol. Validation data are used for parameter selection; Test metrics are not used to select hyperparameters. The public seed-level and validation-search outputs are retained in `results/manuscript/`.
