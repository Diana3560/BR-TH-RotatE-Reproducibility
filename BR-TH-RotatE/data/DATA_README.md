# Dataset notice

This code repository intentionally contains **no dataset files**.

The `data/` directory is documentation-only. The anonymized research dataset is managed and distributed separately from the code repository. No script in this repository automatically locates, downloads, extracts, copies, rewrites, or modifies that separate dataset package.

Data-dependent experiment runners use the file locations declared in `config/multidataset_comparison.yaml`. In the public release those locations point to the ignored `external_data/` workspace, which is empty by design. Researchers who are authorized to run the experiments may provide their own compatible local files there (or adapt the configuration to another local path) without changing the separately distributed dataset package.

Expected dataset-level checksums, counts, modeled relations, and split sizes are retained in the configuration only as reproducibility metadata. They are not a substitute for the separately distributed data.
