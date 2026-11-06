# Model for actinomYcetes Conditional expression using a language model for Natural products (MYClN)

This repository contains the source code release of the final model for predicting gene-level expression in actinomycetes from genomic sequences and culture medium information.

The release is based on the **GenomeBERT + media** configuration from `Final_model/release_ver`.

The model integrates ESM2-35M protein embeddings, genomic context representations from gLM, DNA composition features, and culture medium vectors. It retrieves expression profiles of training genes through k-nearest neighbors (kNN) and generates predictions using an ensemble of eight FiLM heads, LightGBM, and ridge regression stacking.

**Training is performed once, and the trained model is saved for subsequent inference. Prediction commands load the saved model without retraining.**

The gLM checkpoint is `v7_cisfull_bgcoff_best.pth`, with a cis encoder configuration of 96/4.

## Quick Start: Inference with a Pretrained Model

The release has been validated with Python 3.10. Source code and pretrained model artifacts are distributed separately.

**If you have downloaded the pretrained model bundle, you do not need to run `train.py`.**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-preprocessing.txt
```

PyTorch must be installed using a CPU or CUDA wheel compatible with your hardware.

- `requirements.txt` specifies the supported dependency version ranges.
- `requirements-tested.txt` records the direct dependency versions used in the validated A100 environment.

### Downloading Model Artifacts

The pretrained model artifacts are distributed through Hugging Face and should be downloaded into the repository's `artifacts/` directory.

Replace `OWNER/MODEL_REPO` with the actual Hugging Face model repository ID provided by the distributor. This is a placeholder and does not refer to an existing uploaded repository.

```bash
python -m pip install -U huggingface_hub
hf download OWNER/MODEL_REPO --local-dir artifacts
sha256sum -c artifacts/SHA256SUMS
python -c "from model_artifacts import check_model; print(check_model('artifacts/expression'))"
```

For reproducibility, specify a published model tag or commit SHA using `--revision`.

If you received the model as a tar archive, you can extract it instead:

```bash
tar -xf /path/to/Final_release-model.tar
```

See [Hugging Face Upload and Download Guide](docs/HUGGINGFACE.md) for details.

The model artifacts are organized as follows:

- `artifacts/expression/`: kNN retrieval index, prediction heads, LightGBM models, ridge stacking model, and normalization statistics.
- `artifacts/glm/`: gLM checkpoint and input normalization statistics.

The official ESM2-35M weights are downloaded and cached automatically during the first input preparation run.

These artifacts are already available in the existing local working directory.

### Configuration

Inference does not require the original training pickle files or taxonomy CSV files.

If the model artifacts are extracted into their default locations, no `config.local.json` file is required.

To load a model from another directory, specify `--model-dir`.

The reference file paths in `config.local.json` are needed only when reproducing training. See [Training and Inference Guide](docs/SERVING.md).

Relative paths in the JSON configuration are resolved relative to the directory containing the JSON file. Individual paths can also be overridden using environment variables of the form `ACTEXPR_<CONFIG_KEY>`.

## End-to-End Prediction from Raw Genomic Sequences

The pipeline automatically generates the embeddings and input files required for prediction from either a GenBank file or a nucleotide FASTA file, together with culture medium information.

### Step 1. Prepare Model Inputs

```bash
python -m pip install -r requirements-preprocessing.txt

python prepare_inputs.py --genbank genome.gbk \
  --medium ISP2 --out outputs/my_genome --device cuda
```

For a nucleotide FASTA input, use:

```bash
python prepare_inputs.py \
  --fasta genome.fna --topology linear \
  --medium ISP2 --out outputs/my_genome --device cuda
```

When using FASTA input, the pipeline performs CDS prediction before generating the model features.

To specify a custom medium composition, replace `--medium ISP2` with:

```bash
--medium-json examples/medium_composition.json
```

The complete preprocessing workflow is:

**CDS prediction → ESM2-35M embeddings → UniOP operon prediction → promoter/topology features → gLM embeddings → `query.npz`**

The pipeline also generates the following files in the output directory:

- `genome.json`: genome and gene information.
- `medium.json`: encoded culture medium information.
- `query.npz`: model-ready feature arrays.

### Step 2. Run Gene Expression Prediction

```bash
python examples/predict_json.py \
  --model-dir artifacts/expression --device cuda \
  --genome outputs/my_genome/genome.json \
  --medium outputs/my_genome/medium.json \
  --embeddings outputs/my_genome/query.npz \
  --output outputs/my_genome/predictions.json
```

Alternatively, add `--predict` to `prepare_inputs.py` to perform input preparation and expression prediction in a single command.

This option still requires a pretrained expression model, which can be specified using `--model-dir`.

**Model requirements:**

- Input preparation requires gLM weights, gLM normalization statistics, and the ESM model.
- Expression prediction requires only the saved expression model and the prepared query files.
- Original training pickle files and taxonomy files are not accessed during inference.
- If required model artifacts are missing or incomplete, the program exits immediately rather than attempting automatic retraining.

For installation instructions, FASTA gene calling, medium encoding, and offline weight loading, see [Raw Input Guide](docs/RAW_INPUTS.md).

For the input specification when providing precomputed embeddings, see [Input Specification](docs/INPUTS.md).

## Repository Structure

| Path | Description |
|---|---|
| `expression_model.py` | kNN retrieval, FiLM prediction heads, LightGBM, and ridge stacking |
| `train.py` | One-time model training, evaluation, and artifact saving |
| `predict.py`, `serving.py`, `model_artifacts.py` | Saved model validation, loading, and inference on new genomes |
| `prepare_inputs.py` | End-to-end input generation from GenBank/FASTA and culture medium information |
| `config.py`, `configs/head.json` | Portable path configuration and final tuned hyperparameters |
| `schema.py`, `data.py`, `split.py` | Data specifications, transformations, and genus-level splitting |
| `glm/` | Final gLM architecture, pretraining, and embedding extraction |
| `preprocessing/` | CDS prediction, ESM embeddings, UniOP, promoter/topology features, medium encoding, and reference data integration |
| `results/reference/` | Reference evaluation results and confidence intervals |
| `tests/`, `.github/workflows/ci.yml` | CPU integration tests and GitHub Actions CI |
| `artifacts/glm/` | Separately stored gLM weights and normalization statistics; excluded from Git |
| `artifacts/expression/` | Trained expression model artifacts generated by `train.py`; excluded from Git |

## Model Performance and Reproducibility

### Reference Evaluation

The original final evaluation achieved the following results:

| Metric | Value |
|---|---:|
| Test Pearson correlation (r) | **0.637364** |
| Within-condition Pearson correlation (r) | **0.643334** |
| 95% bootstrap confidence interval | **[0.619063, 0.657530]** |

The confidence interval was estimated by bootstrapping the 22 test strains.

The actual saved dataset split contains **160 strains**:

| Dataset split | Number of strains |
|---|---:|
| Training | 133 |
| Validation | 5 |
| Test | 22 |
| **Total** | **160** |

The designation "161-strain" in earlier documentation was an experiment identifier and does not match the total number of strains in the saved evaluation records.

The reported results correspond to the canonical evaluation in which six genera were held out together. They are **not** the mean of leave-one-genus-out (LOGO) evaluations.

### Validation of the Distributed Model

After completing model training and saving the release artifacts, the distributed model achieved:

| Metric | Value |
|---|---:|
| Test Pearson correlation (r) | **0.637365** |
| Within-condition Pearson correlation (r) | **0.643334** |

The bootstrap confidence interval reported above belongs to the original reference evaluation and was not recalculated for the newly saved model.

Inference on an actual genome was also verified by loading the saved model in a new process.

For additional validation details, see [Validation Report](docs/VALIDATION.md).

### Computational Performance

Runtime was measured using *Streptomyces venezuelae* ATCC 10712 grown in ISP2 medium, covering **7,132 genes**, on an NVIDIA A100 80 GB GPU.

| Operation | Runtime |
|---|---:|
| Input preparation | 46.1 s |
| Inference, including model loading and process termination | 10.2 s |
| **Total** | **56.3 s** |
| Prediction function only | 4.1 s |

These measurements were obtained with the model weights and operating system file caches already available.

They should not be interpreted as guaranteed runtimes on other GPUs or as an estimate of minimum GPU memory requirements.

See [Resource Usage](docs/RESOURCE_USAGE.md) for details.

## GitHub Source Code Release

Source code and model artifacts are distributed separately.

Follow the [GitHub Upload Guide](docs/GITHUB_UPLOAD.md) to publish the source code and the [Hugging Face Distribution Guide](docs/HUGGINGFACE.md) to publish model artifacts.

The GitHub repository should contain source code, example configuration files, documentation, and small reference evaluation JSON files.

The `.gitignore` file excludes local configurations, datasets, trained model weights, generated outputs, and caches.

To generate a standalone source distribution ZIP:

```bash
python scripts/package_release.py
# Output: dist/Final_release-source.zip
```

The source ZIP does not include server-specific paths, local configurations, model weights, or raw datasets.

## Pretrained Model Distribution

The trained expression model, kNN reference data, and gLM checkpoint are distributed together through a single Hugging Face Model repository.

The following directories are uploaded without modification:

```text
artifacts/
├── expression/
└── glm/
```

There is no need to split the model artifacts into multiple tar archives.

GitHub hosts the source code, while Hugging Face hosts the model artifacts. The GitHub documentation should identify the Hugging Face model tag or commit corresponding to each source release.

### Preparing Hugging Face Metadata

```bash
python scripts/prepare_hf_metadata.py
# Generates artifacts/README.md and artifacts/SHA256SUMS
# Does not upload files
```

### Optional Offline Model Archive

For offline distribution, the existing packaging script can be used:

```bash
python scripts/package_model.py
# Outputs:
# dist/Final_release-model.tar
# dist/Final_release-model.SHA256SUMS
```

The model archive includes:

**`artifacts/expression/`**

- kNN retrieval index and expression reference table based on 133 training strains
- `reference_genes.tsv`
- Eight trained FiLM prediction heads
- LightGBM models
- Ridge stacking model
- Normalization statistics
- Evaluation results

**`artifacts/glm/`**

- gLM checkpoint
- Input normalization statistics

After extracting the archive into the root directory of a source checkout, inference can be performed directly using `examples/predict_json.py`.

Raw genome input preparation additionally requires the gLM and ESM model weights.

See [Artifact Inventory](docs/ARTIFACTS.md) for a complete description of the distributed files.
