# MYClN artifacts and ActinoBERT identity

This release follows `Final_model/release_ver/config.py`, not the older BGC-on
`Final_model` README. MYClN uses the BGC-free v7 ActinoBERT backbone with
`promoter_conv_dim=96`, `promoter_dils=4`, and the exact head parameters in
`configs/head.json`. Taxonomy recovery from antiSMASH headers is disabled.

## Inputs for one-time expression-model training

Paths below are repository defaults; configure them with `ACTEXPR_CONFIG`.
All pickle files must come from a trusted source: loading Python pickle executes
its serialization instructions. No download URL is invented or bundled.

| Config key | Default relative path | Contract |
|---|---|---|
| `UNIFIED_DATA` | `artifacts/reference/unified_expression_data.pkl` | `gene_wise_data`, `species_metadata`; original labels and 159-d media vectors |
| `ESM_EMB` | `artifacts/reference/embeddings/bert_gene_raw_v5.pkl` | genome → `(genes, 480)` float array |
| `ESM_ORDER` | `artifacts/reference/embeddings/bert_gene_id_order_e2e.pkl` | genome → gene IDs indexing ESM rows |
| `GLM_EMB` | `artifacts/reference/embeddings/bert_gene_embs_v7bgcoff.pkl` | genome → `(genes, 960)` = GIO + CTX |
| `GLM_CLS` | `artifacts/reference/embeddings/bert_cls_vectors_v7bgcoff.pkl` | genome → `(480,)` |
| `GLM_ORDER` | `artifacts/reference/embeddings/bert_gene_id_order_v7bgcoff.pkl` | genome → gene IDs indexing gLM rows |
| `DTU_META` | `data/raw/DTU_Pep_data/DTU_Pep_metadata.csv` | reference DTU taxonomy CSV |
| `MANUAL_META` | `data/raw/Manual_curated_data/Metadata/Actinomycetes_metadata_20260106.csv` | reference manual taxonomy CSV |
| `OPTUNA_HP` | `configs/head.json` | committed final head hyperparameters |

Do not interchange ESM and gLM ordering files. Their gene caps and ordering may
differ. The model retains the source pipeline's independent indexing and zero
features for reference genes absent from an embedding order. Query prediction
requires complete embeddings for all supplied query genes.

## ActinoBERT artifacts

The local delivery includes copies in `artifacts/glm/` of:

- `v7_cisfull_bgcoff_best.pth`: final BGC-free gLM state dictionary.
- `v5_emb_stats.npz`: original input normalization `mean` and `std`.

These files are ignored by Git and excluded from the source ZIP.
Their byte sizes and SHA-256 hashes are in [artifact_manifest.json](artifact_manifest.json).
The pretrained ActinoBERT is an embedding backbone; it is not a serialized expression
retrieval index or fitted FiLM/LightGBM/ridge ensemble.

For optional gLM pretraining, `GLM_WARM_START` points to the BGC-free stage-1
`v6_topo_scratch_bgcoff_best.pth` (not copied into the source distribution).
Raw genome inputs and the stage-1 checkpoint remain external. See
[../preprocessing/README.md](../preprocessing/README.md).

## Configuration and outputs

The original server's `config.local.json` is an ignored, explicit opt-in config.
It reads existing reference artifacts during training. The fitted expression model
defaults to `artifacts/expression/` (`MODEL_DIR`); preprocessing outputs use `outputs/`.
Default `config.py` never refers to that server. Input data are not copied into
this code release. Extraction and preprocessing outputs also default to `outputs/`;
update input config only after reviewing newly generated artifacts.

- `python train.py --check`: existence checks, no large data loaded.
- `python config.py --scope glm`: checkpoint/statistics/species-list existence checks.
- `ACTEXPR_DEVICE=cpu`: small CPU runs; full reference work is designed for CUDA.
- `ACTEXPR_THREADS=1`: limit LightGBM threads for small tests.

The reference expression pipeline holds large tables and retrieval intermediates
in memory during training. A cached A100 inference/preparation run is documented
in [RESOURCE_USAGE.md](RESOURCE_USAGE.md). It is not a minimum-hardware guarantee.

## New-query preparation dependencies

`prepare_inputs.py` only needs the final gLM checkpoint, normalization statistics,
the included `configs/media_catalog.json`, and ESM2-35M weights. The ESM package
automatically uses its cache or downloads the official weights on first use; a
local `--esm-checkpoint` also works. FASTA CDS calling uses Pyrodigal. Install
`requirements-preprocessing.txt`. Large expression-reference files are needed
only when training the expression model, not for prediction or input generation.
Expression prediction needs the exported model directory described in [SERVING.md](SERVING.md).
The complete directory includes training-derived retrieval keys and expression tables,
so it is larger than the neural head weights alone. Keep the directory together when distributing it.
