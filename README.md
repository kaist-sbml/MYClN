# MYClN

**Model for actinomYcetes Conditional expression using a language model for Natural products.**

Predicts gene-level expression in actinomycetes from genomic sequences and culture medium information.

**ActinoBERT** is the genomic language-model backbone. **MYClN** is the complete
expression predictor combining ActinoBERT and ESM-2 representations, DNA features,
culture-medium information, retrieval, and the fitted prediction ensemble.

## Installation

Tested with Python 3.10. Install PyTorch 2.0.1 for your CPU/CUDA environment, then:

```bash
python -m pip install -r requirements-preprocessing.txt
```

## Pretrained Models

Download all `Final_release-model.tar.part-*` files and `SHA256SUMS` from [GitHub Releases](https://github.com/kaist-sbml/MYClN/releases) into a new `downloads/` directory. From the repository root, run:

```bash
(cd downloads && sha256sum -c SHA256SUMS) &&
cat downloads/Final_release-model.tar.part-* | tar -xf - &&
sha256sum -c artifacts/SHA256SUMS
```

This restores `artifacts/expression/` and `artifacts/glm/`. ESM2 weights are downloaded automatically on first use. No retraining is required.

## Prediction

```bash
python prepare_inputs.py --genbank genome.gbk \
  --medium ISP2 --out outputs/my_genome --device cuda --predict
```

Predictions are saved to `outputs/my_genome/predictions.json`. Use a new output directory for each run.
For CPU execution, use `--device cpu`.
For FASTA input, replace `--genbank genome.gbk` with `--fasta genome.fna --topology linear` (or `circular`, as appropriate).
For a custom medium, replace `--medium ISP2` with `--medium-json examples/medium_composition.json`.

## Documentation

- [Input preparation](docs/RAW_INPUTS.md)
- [Training and inference](docs/SERVING.md)
- [Validation and performance](docs/VALIDATION.md)
- [Model artifacts](docs/ARTIFACTS.md)
