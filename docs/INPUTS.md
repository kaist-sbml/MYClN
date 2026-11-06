# Query input contract

For GenBank/FASTA plus a medium name or composition, run `prepare_inputs.py`; it
automatically generates the files described below, including the internal genome ID. See [RAW_INPUTS.md](RAW_INPUTS.md).

`Genome` contains `genome_id`, `organism`, and `genes`, with each gene carrying
`gene_id`, `aa_sequence`, `dna_sequence`, and optional promoter/coordinate fields.
Use a new genome ID, unique gene IDs, and at least one gene. IDs belonging to the
training retrieval corpus are rejected; the raw-input CLI generates IDs automatically.

`Medium` contains `medium_id` and a finite 159-element `vector` in the reference
component order. The raw-input CLI encodes ingredient names and amounts using the bundled reference catalog. The optional
`composition` and `params` fields are descriptive and are not automatically
encoded by `predict()`. Keep the original expression-label transformation;
outputs are in the reference label space, not raw TPM.

For `predict(genome, medium, new_embeddings)`, each embedding entry maps the
query genome ID to its array/list:

| Entry | Shape/content |
|---|---|
| `esm_emb` | `(n_esm, 480)` raw ESM2-35M embeddings |
| `esm_order` | `n_esm` gene IDs, explicit ESM row ordering |
| `glm_emb` | `(n_glm, 960)` concatenated GIO + CTX |
| `glm_order` | `n_glm` gene IDs, explicit gLM row ordering |
| `glm_cls` | `(480,)` genome-level CLS vector |

Each order must contain every requested gene once. ESM and gLM row orders can
differ; never assume they match. Prediction returns `{gene_id: float}` for the
requested medium. Training statistics and fitted weights remain fixed. Query genes
are searched against the saved training-only index; query labels are never needed.

For `examples/predict_json.py`, provide Genome and Medium as JSON objects with the
same field names, and a single NPZ with the five entries above (arrays directly,
without the outer genome-ID mapping). Store the two order arrays as NumPy string
arrays, not object arrays: the CLI uses `allow_pickle=False`.

Pass `model_dir="artifacts/expression"` (the default) and optionally `device="cuda"`
to `predict()`. The model must first be exported by `train.py`; missing or incomplete
models fail without training. No original reference pickle or taxonomy CSV is read.
For multiple requests, reuse `serving.ExpressionPredictor(model_dir, device=...)`.
See [SERVING.md](SERVING.md). Use `prepare_inputs.py` to produce the query features
from a new genome sequence.
