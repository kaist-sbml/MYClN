# Preprocessing and gLM reproduction

For a new genome, use the complete raw-input pipeline:

```bash
python -m pip install -r requirements-preprocessing.txt
python prepare_inputs.py --genbank genome.gbk \
  --medium ISP2 --out outputs/my_genome --device cuda
```

It reads CDSs (or calls them from nucleotide FASTA), runs ESM2-35M, infers operons
with the original UniOP-style recipe, builds promoter/topology features, runs the
final gLM, and writes genome.json + medium.json + query.npz. The frozen reference
media vocabulary and named recipes are included. See [raw-input instructions](../docs/RAW_INPUTS.md).

The sections below describe the additional **historical full reference dataset
rebuild**, corpus extraction and optional gLM pretraining workflows. These retain
their original external raw collection layouts. New-query preparation does not
need those collection trees or precomputed query embeddings.

## Dataset assembly

Set `RAW_DATA_DIR` and `TOOLS_DIR` in the JSON configuration. The expected layout
under `RAW_DATA_DIR` is:

- `DTU_Pep_data/`: GenBank files in `antismash_results`, `log_tpm`, `Media.csv`,
  `Whole_genome_inputs` operon/promoter files, and original annotation assets.
- `Manual_curated_data/`: `Genome`, `TPM`, `Metadata`, `Whole_list`, plus the
  original Pfam/EC assets used by the source dataset builder.
- `Whole_genome_data/`: `cds_embeddings`, operon TSV, promoter FASTA.

`python -m preprocessing.build_dataset` builds the internal table at
`outputs/unified_expression_data.pkl` and a schema dataset under `outputs/dataset`.
`BUILD_INTERNAL_PKL` and `DATASET_DIR` override those outputs. The producer keeps
its historical source layout, including antiSMASH-named GenBank directories and
vocabulary files. BGC is unused by the released expression predictor, but a raw
rebuild still needs the input files referenced by this historical producer.
Its `MediaVectorizer` defines component order; preserve the reference vocabulary
when reproducing the original model. A newly built vocabulary can define a
different model and is not automatically compatible with the 159-d query API.

## Genome inputs

`python -m preprocessing.genome_features` creates `operon_meta.npz` in each
prepared genome directory from `cds_emb.npy`, `cds_meta.json`, the operon table,
and promoter FASTA. This is an explicit producer that writes into the configured
raw genome corpus. Run it on a working copy if preserving existing features.

Each gLM genome directory needs:

- `cds_emb.npy`: protein embeddings from external `esm2_t12_35M_UR50D`, 480-d.
- `cds_meta.json`: CDS metadata and stable gene identifiers.
- `operon_meta.npz`: operon assignments, strand, spacing, promoter one-hot and
  numeric features, valid masks, sizes, global features.
- `topology_meta.npz`: optional per-operon contig metadata. The source extractor
  falls back to one circular contig when absent; this is an assumption, not
  inferred topology.

Search roots include DTU, the whole-genome corpus, and the manual collection.
Override `DTU_GENOME_ROOT`, `GENOME_CORPUS`, and `MANUAL_GENOME_ROOT` as needed.

## Extract embeddings

```bash
python config.py --scope glm
python -m glm.extract
```

Extraction requires `GLM_CHECKPOINT`, `EMB_STATS`, and `SPECIES_ORDER` (a gene-order
pickle whose keys select genomes). It writes `bert_gene_{embs,raw,id_order}_v7bgcoff.pkl`
and `bert_cls_vectors_v7bgcoff.pkl` under `outputs/embeddings/`.
The raw output is unstandardized ESM; normalization is applied only to gLM inputs.
`EXTRACT_OUTPUT_DIR` changes the output directory and `GLM_OUT_SUFFIX` the suffix.
A new genome can be selected with a pickle mapping its genome ID to its gene IDs.

The extractor loads the entire checkpoint with `strict=True`. Missing genomes,
empty gene selections, or extraction errors fail before writing aggregate pickles.
The historical caps (13,000 genes, 9,000 operons) remain; inspect coverage before
using genomes above these limits.

`python -m preprocessing.esm_embeddings` can separately collect the protein
channel from existing `cds_emb.npy` files. It aligns rows by explicit gene IDs in
`SPECIES_ORDER` and fails on missing/misaligned inputs. It does not run ESM-2.

## Optional pretraining

```bash
python -m glm.pretrain
```

This long CUDA job uses the BGC-free cis+topology recipe and external BGC-free
stage-1 checkpoint at `GLM_WARM_START`. It writes under `outputs/glm/` by default.
Importing `glm.pretrain` does not start a job. The warm-start helper intentionally
allows shape-matching partial loads during training; extraction requires a full
match. Existing run checkpoints reload weights only, not optimizer/scaler/epoch
state, so this is not an exact interrupted-training resume.

Environment overrides for research use: `GLM_TOPO_ONLY=1`, `GLM_FROM_SCRATCH=1`,
`GLM_EPOCHS`, `GLM_CKPT_NAME`. Defaults preserve the final cis model dimensions,
one epoch, and BGC loss off. Producing new embeddings or changing the pretraining
recipe requires reevaluating the downstream model.
