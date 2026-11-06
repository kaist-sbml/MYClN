# Validation record — 2026-11-05

## Existing reference results

Copied unchanged from `Final_model/release_ver/trained_model/`:

- [result.json](../results/reference/result.json): pooled Pearson r
  `0.6373644471168518`, within-condition r `0.6433338686865021`,
  `1,233,774` test rows, exact split membership and tuned head parameters.
- [result_ci.json](../results/reference/result_ci.json): species-block 95% CI
  `[0.6190632743558973, 0.6575303190912413]`, 5,000 bootstrap resamples,
  22 test strains across six genera.

The stored split lists have **133 train + 5 validation + 22 test = 160 entries**.
Earlier prose called this the “161-strain” experiment and even described it as
LOGO. This release reports the saved lists and canonical held-out-genus protocol
explicitly. We have not rerun full-data training to diagnose or revise historical
coverage, and do not substitute the older BGC-on model's 0.6424 score.

## Checks performed for this code release

Environment: Linux, Python 3.10.20, exact direct package versions in
[requirements-tested.txt](../requirements-tested.txt).

- Python source compilation and command-line help checks.
- Seven `unittest` checks: path resolution from a different working directory,
  schema serialization, genus/genome leakage guards, consistent taxonomy recovery
  configuration, query collision/alignment validation, import safety, and the
  actual end-to-end expression stack on synthetic data.
- The synthetic stack uses 100 training, 24 validation and 24 test genes, each
  under two media. It runs retrieval, validation k/temperature selection, one
  small FiLM seed for two epochs, LightGBM and ridge stacking. All predictions
  are finite. Permuting gLM rows together with its own order preserves features;
  replacing test labels preserves the final predictions.
- Comparison with the original `release_ver/expression_model.py` on the same
  synthetic input: identical feature tables, selected retrieval k/temperature,
  and final predictions; maximum prediction difference **0.0**. The original
  was adapted only to use CPU float32 retrieval matmul and one LightGBM thread
  for this comparison, matching the release CPU path. CUDA retrieval remains
  the original fp16 path.
- All configured local training inputs and gLM inputs exist.
- The actual final BGC-free checkpoint loads with `strict=True`: all keys match,
  **12,858,363 parameters**, global input dimension 12.
- A forward pass with that checkpoint on a synthetic eight-gene/two-operon input
  produces finite GIO/CTX `(1, 8, 480)` and CLS `(1, 480)` tensors.
- Feature preprocessing on existing `NBC_00001` inputs succeeds with 9,830 genes
  and 6,922 operons; the resulting gene input has shape `(1, 9830, 480)` and is finite.

The source ZIP is checked for machine-specific home paths, symlinks and large
files. Local config, weights, dataset pickles and generated outputs are excluded.
The source is also tested after extraction into a fresh temporary directory.
The GitHub Actions workflow is included; hosted GitHub CI has not been executed.

## Limits of this validation

These are packaging and smoke checks, not a new scientific evaluation. Full
reference training, full-genome embedding extraction over the corpus, all LOGO
folds, raw dataset reconstruction and CUDA numerical reproducibility were not
rerun. The historical reference metrics are carried forward with provenance.
No claim is made that newly rebuilt data/vocabularies reproduce those metrics.
Expression training and serving are now separate. The initial fitted-model export checks used synthetic data. The subsequent
full-data export and real-genome inference are recorded below. The existing gLM checkpoint is not a fitted expression model.

## Raw-input preparation validation

The added preparation path was tested with actual ESM2-35M weights and the actual
final BGC-free gLM checkpoint on CPU, without substituting mock embeddings:

| Input | CDSs | Operons | Final gLM shape | Result |
|---|---:|---:|---|---|
| Synthetic annotated GenBank | 64 | 32 | `(64, 960)` | Complete query bundle generated |
| Same nucleotide FASTA, Pyrodigal meta mode | 65 | 32 | `(65, 960)` | Complete query bundle generated |

Both produce finite ESM `(N, 480)`, gLM `(N, 960)`, CLS `(480,)`, explicit gene
orders, and a 159-dimensional medium vector. Different CDS counts are expected
when calling genes from sequence versus using supplied annotations. The examples
are synthetic pipeline checks, not expression-accuracy benchmarks.

The raw-input addition brought the suite to **15 tests**, including FASTA gene calling, strand-aware DNA
extraction, UniOP determinism and coverage, failure on underspecified fragment
inputs, named/composition media equivalence, ambiguous-name rejection and ESM
cleaning/cropping. All operon/promoter/global arrays in the synthetic GenBank
example match the original `genome_features.py` producer exactly.

The catalog's 159-component ordering was checked against the actual unified
reference pickle. The 34 included named media matched reference-condition
vectors sampled across its genomes; names with observed alternative recipes
are explicitly marked ambiguous. The catalog does not regenerate a new
vocabulary from user query data.

Additional tested packages: fair-esm 2.0.0 and Pyrodigal 3.7.1. The latter was
installed in an isolated temporary dependency directory during local validation.
Full reference expression prediction after raw preprocessing was not rerun; the
prepared query contract and expression stack were tested separately.

## Persisted expression-model validation

The complete suite now has **21 tests**. All passed locally, including the CUDA
export/reload test on the available A100; that test is skipped on CPU-only CI.
The six additional serving checks cover:

- Training and exporting two small FiLM seeds, LightGBM and ridge, then reloading
  predictions for both test media. CPU results agree within `rtol=1e-5, atol=1e-5`.
  The fixture has 100 training genes across two genomes, 24 validation genes and
  24 test genes; using multiple training genomes exercises CLS normalization.
- Predictions while reference loading, cache construction, head training and
  optimizer construction are patched to raise if called.
- Independent ESM/gLM row permutations, reversed query order, a one-gene subset,
  and tiled retrieval with small query/key tiles.
- An unseen medium versus the original retrieval fallback calculation.
- A fresh-process JSON CLI run from a different working directory with nonexistent
  reference-data paths; only exported model files and query files are available.
- Missing-model failure before torch/input loading, a truncated-model rejection,
  and CUDA fp16 export/reload plus CPU loading of a CUDA-exported model.

The observed maximum CUDA prediction difference in the small export/reload check
was approximately **0.000264**. A diagnostic comparison found identical head input
features and a one-fp16-ULP similarity difference in the dual index when GEMM was
run with query-only rows, with some near-tied neighbors changing order. The CUDA
check allows `rtol=5e-4, atol=5e-4`; bitwise equivalence across devices/batch shapes
is not claimed. This is a numerical check, not an accuracy measurement.

The original research model comparison was also rerun after adding persistence:
CPU synthetic feature tables, selected k/temperature and predictions remained
identical, with maximum prediction difference **0.0**.

The local train-once wrapper passes all reference path checks. The real-data
prediction wrapper exits immediately with a one-time-training instruction because
no complete expression model has yet been exported. Existing `prepared_ISP2`
inputs are retained. No full-data training or S. albidoflavus expression prediction
was started as part of this refactor, and no whole-model latency claim is made.

## Exported expression model — 2026-11-05

`python train.py --output artifacts/expression` was run once on the full reference
data (CUDA, A100 80GB) and the fitted model was exported.

- Split 133/5/22 identical to the reference `result.json`; head HP identical.
- Test Pearson r `0.6373652`, within-condition r `0.6433341` (reference `0.6373644`, `0.6433339`).
  Row keys and measured values of `test_predictions.npz` are identical to the reference run;
  per-row prediction max |diff| `1.2e-3`, correlation `0.99999999` (CUDA nondeterminism).
- Reload: `serving.ExpressionPredictor` on the exported directory, all 22 test genomes,
  their media (306,368 gene×medium predictions), reproduces the training-time predictions
  within `6.4e-4` without reading reference data or fitting.
- `reference_genes.tsv` (1,038,924 rows) was rebuilt with `scripts/export_reference_genes.py`
  and checked against `keys_raw.npy` (2,000 sampled rows, min cosine 1.00000).
- `scripts/package_model.py` bundle (`dist/Final_release-model.tar`, 9.8 GB, 28 files + SHA256SUMS)
  passes `sha256sum -c` after extraction.


## Completed model and real-genome inference — 2026-11-06 update

A complete real-data export is now present in the local `artifacts/expression/`.
All 22 required serving files pass the completion-manifest check: four reference
indexes, expression tables/statistics, eight fitted heads, LightGBM and ridge.
The reference index contains 1,038,924 training genes. The saved run reports
pooled test Pearson r **0.6373652219772339** and within-condition r
**0.6433341271048655** on 1,233,774 test rows. Historical confidence intervals
were not recalculated for this export.

Fresh-process prediction using the complete saved model succeeded for both
S. albidoflavus J1074 / ISP2 (5,832 genes) and S. venezuelae ATCC 10712 / ISP2
(7,132 genes). Outputs contain every requested gene and finite values.
The latter was also rerun with resource tracking; see [RESOURCE_USAGE.md](RESOURCE_USAGE.md).
These are workflow checks on example genomes, not a new held-out accuracy benchmark.

Source packaging checks were repeated after updating the saved-model quick start
and manual GitHub upload instructions. Model files, local configuration, downloads,
logs and generated outputs remain excluded from the Git/source distribution.
