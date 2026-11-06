#!/usr/bin/env python
"""Protein (ESM-2) per-gene embeddings — the retrieval RAW / protein channel.

Collects the raw external ESM-2 per-CDS embeddings (`cds_emb.npy`, 480-d) for every
genome and re-orders each genome's rows to match, exactly, the per-gene order the
gLM feature stage emitted (GENE_ORDER_PKL). This guarantees the protein channel
lines up row-for-row with the gLM channel inside the retrieval model.

Inputs (per genome directory, searched across ROOTS):
  cds_emb.npy       (G0, 480)   raw ESM-2 per-CDS embeddings (external ESM-2 pass)
  operon_meta.npz               operon layout / promoter / masks  (genome_features.py)
  cds_meta.json     list[{locus_tag/gene_id, contig, start, end, strand}]
Global input:
  GENE_ORDER_PKL    {genome: [gene_id]}   canonical per-gene order (from the gLM extract)

Output:
  ESM_EMB_PKL       {genome: (G, 480)}    rows aligned to GENE_ORDER_PKL[genome]

The ESM-2 model used to produce cds_emb.npy is esm2_t12_35M_UR50D (480-d).
This step is NOT runnable without the external ESM-2 embeddings + the gene-order
pickle produced by the gLM feature extraction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as release_config
import os, json, pickle
import numpy as np

# ── config: real artifact paths (configured via config.py — renaming breaks them) ─────────
SAVE_DIR = release_config.EXTRACT_OUTPUT_DIR
ROOTS = release_config.GENOME_ROOTS
GENE_ORDER_PKL = release_config.SPECIES_ORDER
ESM_EMB_PKL = os.path.join(SAVE_DIR, 'bert_gene_raw_v5.pkl')

# Gene / operon caps — MUST match the gLM feature extraction that built GENE_ORDER_PKL.
MAX_OPERONS = 9000
MAX_GENES   = 8092


def find_genome_dir(genome: str):
    """First ROOTS/<genome> that has all three required files."""
    for r in ROOTS:
        gd = os.path.join(r, genome)
        if all(os.path.exists(os.path.join(gd, f))
               for f in ["cds_emb.npy", "operon_meta.npz", "cds_meta.json"]):
            return gd
    return None


def surviving_indices(cds_emb, meta):
    """Reproduce the gLM extractor's gene filtering -> surviving original indices.

    Mirrors the gene-selection the gLM feature stage applies (truncate to the shared
    length, keep genes whose operon index < number of kept operons, cap at MAX_GENES),
    so the returned rows correspond to GENE_ORDER_PKL[genome] one-for-one.
    """
    operon_id = meta["operon_id_of_gene"].astype(np.int64)
    strand    = meta["gene_strand"].astype(np.int64)
    inter     = meta["intergenic_upstream"].astype(np.float32)
    prom_oh   = meta["promoter_onehot"]

    orig = np.arange(len(cds_emb))
    n = min(len(cds_emb), len(operon_id), len(strand), len(inter))
    operon_id = operon_id[:n]; orig = orig[:n]

    O = min(len(prom_oh), MAX_OPERONS)
    if O > 0:
        keep = operon_id < O
        operon_id = operon_id[keep]; orig = orig[keep]
    else:
        orig = orig[:0]
    G = min(len(orig), MAX_GENES)
    return orig[:G]


def main():
    gene_order = pickle.load(open(GENE_ORDER_PKL, "rb"))
    genomes = sorted(gene_order.keys())
    print(f"genomes: {len(genomes)}")

    raw_embs = {}
    mismatch, missing = [], []
    for genome in genomes:
        gd = find_genome_dir(genome)
        if gd is None:
            missing.append(genome); continue
        cds = np.load(os.path.join(gd, "cds_emb.npy")).astype(np.float32)
        with open(os.path.join(gd, 'cds_meta.json')) as handle:
            cds_meta = json.load(handle)
        gid_all = [g.get('locus_tag', g.get('gene_id', str(k))) for k, g in enumerate(cds_meta)]
        positions = {gid: i for i, gid in enumerate(gid_all)}
        ids = list(gene_order[genome])
        if len(positions) != len(gid_all) or any(gid not in positions for gid in ids):
            raise ValueError(f'Cannot align protein embeddings to gene order for {genome}')
        raw = cds[[positions[gid] for gid in ids]]
        raw_embs[genome] = raw

    if missing:
        raise FileNotFoundError(f"Missing genome inputs: {missing}")
    if not raw_embs:
        raise ValueError("No protein embeddings were collected")
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(ESM_EMB_PKL, "wb") as f:
        pickle.dump(raw_embs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved {ESM_EMB_PKL} | genomes={len(raw_embs)} | "
          f"missing={len(missing)} | order_mismatch={len(mismatch)}")
    if mismatch:
        print("  mismatch(first5):", mismatch[:5])
    g0 = next(iter(raw_embs))
    print(f"  sample {g0}: raw{raw_embs[g0].shape}  | "
          f"ids={len(gene_order[g0])} (rows==ids? {raw_embs[g0].shape[0]==len(gene_order[g0])})")
    print(f"  dim set: {set(v.shape[1] for v in raw_embs.values())}")


if __name__ == "__main__":
    main()
