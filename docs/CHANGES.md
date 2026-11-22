# Changes from the final research release

Source file identities and original SHA-256 digests are recorded in
[source_manifest.json](source_manifest.json). These identify the original source snapshots; subsequent naming updates are recorded below.

- Selected the final `release_ver` BGC-free configuration and copied its exact
  tuned head parameters and measured results. Excluded old model variants,
  exploratory notebooks, paper drafts, figures, logs, caches and backup outputs.
- Replaced machine-specific executable paths with shared repository-relative
  defaults, explicit JSON configuration and environment overrides. A separate
  ignored local config connects to the existing research data.
- Added CLI help, input existence checks and output-overwrite control to training;
  added a JSON/NPZ query example CLI, requirements and GitHub Actions checks.
- Canonical split preparation uses the explicit antiSMASH-taxonomy setting.
  Prediction now consumes the saved training-only index without recomputing splits.
- Rejected query/reference ID collisions, incomplete query embeddings, invalid
  media dimensions and ambiguous gene ordering. Protein feature collection uses
  explicit gene-ID alignment, including manual-curated search roots.
- Kept the CUDA model arithmetic, final hyperparameters and architecture intact.
  Added float32 retrieval matmul for small CPU tests. LightGBM errors now fail
  visibly instead of silently producing an incomplete ensemble.
- Made gLM pretraining safe to import. Extraction requires a complete checkpoint
  match and reports missing/failed genomes before writing aggregate results.
  The stage-2 warm-start default points to the BGC-free stage-1 backbone.
- Separated expression training from inference: train.py exports all four retrieval
  indexes, training-only expression tables/statistics, accepted seed heads, LightGBM
  and the fitted ridge. predict.py only loads the resulting model. The gLM checkpoint
  remains a separate embedding backbone.
- Corrected the public evaluation description to the actual saved 133/5/22 split
  and canonical six-genus holdout. Existing result JSON values are unchanged.

Local and source-archive delivery are separate: `artifacts/glm/` contains copies
of the final backbone weights and normalization statistics for this workspace;
Git and the ZIP exclude them. Their hashes are recorded in
[artifact_manifest.json](artifact_manifest.json).

## Raw-input pipeline added

- Added `prepare_inputs.py` from GenBank or nucleotide FASTA and named/composition
  medium to all query files, optionally followed by expression prediction.
- Ported ESM2-35M cleaning/cropping/pooling and UniOP-style KDE inference from the
  original Tools notebooks. Added promoter and topology producers with explicit
  gene membership and the same numerical feature recipe.
- Added optional Pyrodigal gene calling for FASTA, frozen reference medium
  encoding, synthetic examples, provenance manifests and additional tests.
- Included 159 reference components, 34 verified unambiguous named media, and
  explicit rejection of conflicting medium names or unknown ingredients.

The raw-input CLI now generates its internal query identifier automatically from
the input file. Standard commands only require input, medium and output; the old
`--genome-id` remains an optional compatibility override.

## Fitted expression serving added

- Export a versioned model with an atomic completion manifest; incomplete exports
  fail before query loading, with an explicit one-time training command.
- Memory-map reference keys and transfer bounded key tiles to the device. Compute
  neighbors only for query genes, never the reference-versus-reference cache.
- Freeze feature/media/retrieval normalization, selected k/temperature, prior grid,
  head weights and ridge normalization. Preserve the original feature/stack order.
- Support reusable predictors and different devices. Validate saved-model versus
  in-memory predictions, permuted orders, query subsets, unseen media, fresh-process
  use without reference files, truncated exports and CUDA export/reload.
- Update the JSON CLI and raw-input --predict path to use MODEL_DIR or --model-dir.
  Previous reference/hp arguments to predict() are removed: fitting belongs in train.py.

## Model naming — 2026-11-22

- The final genomic backbone is named **ActinoBERT**; the complete expression predictor is **MYClN**.
- `build_model()` returns `ActinoBERT`. `GenomeBERT` remains an alias for compatibility.
- Checkpoint filenames, state-dictionary keys, environment variables, artifact formats, and evaluation values are unchanged.
