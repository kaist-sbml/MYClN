---
tags:
- biology
- genomics
- gene-expression
- retrieval
- pytorch
---

# Actinomycete gene-expression model — BGC-free gLM + media

Fitted model artifacts for predicting per-gene expression from an actinomycete
genome and a growth-medium vector. The accompanying GitHub source provides the
GenBank/FASTA input preparation and inference code.

**Publisher:** add the GitHub source URL and select the applicable model/data
license before publishing this card. No license is assigned by this template.
This is a custom PyTorch/LightGBM retrieval pipeline, not a Transformers AutoModel.

## Files and model identity

- `glm/`: `v7_cisfull_bgcoff_best.pth` and its input normalization statistics.
  The backbone uses cis encoder dimension 96 and dilation setting 4.
- `expression/heads/`: eight fitted FiLM head state dictionaries.
- `expression/keys_*.npy`: full, dual, raw and GIO retrieval indexes.
- `expression/train_mean.npy`, `expression.npy`, `has_expression.npy`,
  `media_vectors.npy`: training-reference means and medium-matched expression tables.
- `expression/feature_stats.npz`, `transforms.npz`: fitted feature normalization.
- `expression/lightgbm.txt`, `stack.npz`: fitted LightGBM and ridge stacking.
- `expression/manifest.json`, `fit_metadata.json`, `reference_metadata.json`:
  model format, selected retrieval parameters and reference metadata.
- `expression/reference_genes.tsv`: reference index row-to-gene mapping.
- `expression/result.json`, `test_predictions.npz`: export-run evaluation results.
- `SHA256SUMS`: file integrity checks, using paths relative to the GitHub checkout.

Keep the indexes, expression tables, statistics and heads from the same revision.
The reference tables contain training-derived expression measurements and features;
this repository contains more than neural-network weights alone.

## Use with the GitHub source

After installing the source dependencies and `huggingface_hub`, run from the
source checkout root. Replace `OWNER/MODEL_REPO` with this model repository ID.

```bash
hf download OWNER/MODEL_REPO --local-dir artifacts
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"

python prepare_inputs.py --genbank genome.gbk --medium ISP2 \
  --out outputs/my_genome --device cuda
python examples/predict_json.py \
  --model-dir artifacts/expression --device cuda \
  --genome outputs/my_genome/genome.json \
  --medium outputs/my_genome/medium.json \
  --embeddings outputs/my_genome/query.npz \
  --output outputs/my_genome/predictions.json
```

For reproducible use, add `--revision` with the published release tag or commit SHA.
The ESM2-35M protein backbone is downloaded separately by the input-preparation
library on first use, or supplied with `--esm-checkpoint`. No expression retraining
or original reference pickle/taxonomy CSV is needed for inference.

## Evaluation and limitations

The saved export reports pooled test Pearson r **0.637365** and within-condition
r **0.643334** on **1,233,774 test rows**. The canonical split has **133 training,
5 validation and 22 test genomes**, holding out six genera together. This is not
an average over leave-one-genus-out folds. Outputs are in the reference expression
label space, not raw TPM.

A single cached A100 80GB run on S. venezuelae ATCC 10712 / ISP2 (7,132 retained
genes) took 46.1 seconds for preparation and 10.2 seconds for prediction including
process/model loading. These are workflow timings, not independent accuracy
measurements or minimum-hardware guarantees. See the source documentation for
raw-input conventions, validation provenance and resource details.
