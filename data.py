"""Adapter between the standard input schema (schema.py) and the tensors the
retrieval model consumes.

Two directions:
  from_schema(dataset)      Dataset            -> internal expression table  (for a fresh build)
  to_schema(internal, ...)  internal table     -> Dataset                    (export existing data)

Plus the reference loader used by train.py/predict.py:
  load_reference(cfg)       -> everything build_cache() needs (table + embeddings + orders)

The model needs precomputed per-gene embeddings (protein=ESM-2, genomic=gLM). Those are
produced once by the feature stage (glm/extract.py + an ESM pass) and referenced by path;
they are NOT re-derived here.
"""
import pickle, numpy as np
from schema import Genome, Gene, Medium, ExpressionRecord, Dataset


# ── internal expression table = {"gene_wise_data": {...}, "species_metadata": {...}} ──
# gene_wise_data[genome_id][gene_id] = {
#   "gene_info": {gene_sequence_aa, gene_sequence_dna, promoter_sequence, ...},
#   "conditions": [{media_name, media_vector, expression}, ...],
#   "source": str }

def from_schema(dataset: Dataset) -> dict:
    """Standard-schema Dataset  ->  internal expression table consumed by build_cache."""
    med = {m.medium_id: m for m in dataset.media}
    # gene -> list of (medium_id, expression)
    obs = {}
    for r in dataset.expression:
        obs.setdefault((r.genome_id, r.gene_id), []).append((r.medium_id, r.expression))

    gwd, smeta = {}, {}
    for gm in dataset.genomes:
        smeta[gm.genome_id] = {"organism": gm.organism, "source": "schema"}
        genes = {}
        for gene in gm.genes:
            conds = []
            for mid, expr in obs.get((gm.genome_id, gene.gene_id), []):
                m = med[mid]
                mv = np.asarray(m.vector, np.float32) if m.vector is not None \
                     else _vectorize(m.composition)
                conds.append({"media_name": mid, "media_vector": mv, "expression": float(expr)})
            genes[gene.gene_id] = {
                "gene_info": {"gene_sequence_aa": gene.aa_sequence,
                              "gene_sequence_dna": gene.dna_sequence,
                              "promoter_sequence": gene.promoter_sequence,
                              "bgc_id": None},
                "conditions": conds, "source": "schema"}
        gwd[gm.genome_id] = genes
    return {"gene_wise_data": gwd, "species_metadata": smeta}


def to_schema(internal: dict) -> Dataset:
    """Existing internal table  ->  standard-schema Dataset (for export / inspection).
    Media are emitted with their model-ready vector (composition left empty — the raw
    ingredient map is not stored in the internal table)."""
    gwd = internal["gene_wise_data"]; smeta = internal.get("species_metadata", {})
    genomes, media, expr = [], {}, []
    for gid, genes in gwd.items():
        organism = smeta.get(gid, {}).get("organism", "")
        glist = []
        for gene_id, gd in genes.items():
            gi = gd["gene_info"]
            glist.append(Gene(gene_id=gene_id, aa_sequence=gi.get("gene_sequence_aa", ""),
                              dna_sequence=gi.get("gene_sequence_dna", ""),
                              promoter_sequence=gi.get("promoter_sequence")))
            for c in gd["conditions"]:
                mid = c["media_name"]
                if mid not in media:
                    v = c.get("media_vector")
                    media[mid] = Medium(medium_id=mid,
                                        vector=(np.asarray(v).tolist() if v is not None else None))
                if c.get("expression") is not None:
                    expr.append(ExpressionRecord(gid, gene_id, mid, float(c["expression"])))
        genomes.append(Genome(genome_id=gid, organism=organism, genes=glist))
    return Dataset(genomes=genomes, media=list(media.values()), expression=expr)


def _vectorize(_composition):  # placeholder: a real MediaVectorizer maps ingredients -> the fixed media space
    raise NotImplementedError(
        "Medium.vector is None and no MediaVectorizer is configured. Provide Medium.vector "
        "(the model-ready fixed-dim media vector) or fit a vectorizer from the reference media.")


# ── reference loader (the artifacts the deployed model was built from) ──
def load_reference(cfg):
    """Load the internal table + precomputed embeddings for training/inference."""
    from pathlib import Path
    required = ("UNIFIED_DATA", "ESM_EMB", "ESM_ORDER", "GLM_EMB", "GLM_CLS", "GLM_ORDER")
    missing = [getattr(cfg, name) for name in required if not Path(getattr(cfg, name)).is_file()]
    if missing:
        raise FileNotFoundError("Missing reference artifacts: " + ", ".join(missing))
    def L(p):
        with open(p, "rb") as handle:
            return pickle.load(handle)
    return dict(unified_data=L(cfg.UNIFIED_DATA),
                esm_emb=L(cfg.ESM_EMB), esm_order=L(cfg.ESM_ORDER),
                glm_emb=L(cfg.GLM_EMB), glm_cls=L(cfg.GLM_CLS), glm_order=L(cfg.GLM_ORDER))


def reference_split(reference, cfg):
    """Use identical canonical taxonomy and embedding coverage for train/predict."""
    import split
    uni = reference['unified_data']
    tax = split.build_taxonomy(uni, cfg.DTU_META, cfg.MANUAL_META, cfg.ANTISMASH_DIR)
    mby = {sp: {c.get('media_name') for c in next(iter(genes.values()))['conditions']}
           for sp, genes in uni['gene_wise_data'].items() if genes}
    fold = split.make_split(tax, mby)
    cov = set(reference['esm_emb']) & set(reference['glm_emb'])
    return {name: [s for s in getattr(fold, name + '_species') if s in cov]
            for name in ('train', 'val', 'test')}
