"""Standard input schema for the gene-expression model.

The whole package is built around TWO inputs and one label:

    Genome  ──┐
              ├──►  expression(gene, medium)          (prediction)
    Medium  ──┘

    ExpressionRecord = observed expression of one gene of one genome in one medium
                                                                    (training label)

Everything upstream (preprocessing) produces objects conforming to this schema;
everything downstream (feature extraction, retrieval model, training, inference)
consumes only this schema. There is no other input contract.

On-disk canonical form (a single directory `dataset/`):
    dataset/
      genomes.pkl      list[Genome]         (sequences + gene layout per organism)
      media.pkl        list[Medium]         (medium composition)
      expression.pkl   list[ExpressionRecord]   (training labels; omitted at inference)
Load/save with `load_dataset(dir)` / `save_dataset(ds, dir)` below.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import os, pickle


# ─────────────────────────────── Genome ───────────────────────────────
@dataclass
class Gene:
    """One protein-coding gene. Sequences are the ONLY per-gene requirement;
    contig/strand/position let the pipeline reconstruct operon structure."""
    gene_id: str
    aa_sequence: str                    # protein sequence  → ESM + gLM embedding
    dna_sequence: str                   # CDS nucleotide seq → GC / codon composition
    promoter_sequence: Optional[str] = None   # upstream region → cis features (gLM pretrain)
    contig_id: str = "0"                # replicon/contig this gene sits on
    strand: int = 1                     # +1 / -1
    start: int = 0                      # gene start coordinate on the contig (bp)
    end: int = 0                        # gene end coordinate


@dataclass
class Genome:
    """One organism's genome = an ordered set of genes + a taxonomic label.
    `organism` (binomial, e.g. 'Streptomyces coelicolor') is the ONLY thing the
    leakage-safe genus split needs."""
    genome_id: str
    organism: str
    genes: list[Gene] = field(default_factory=list)


# ─────────────────────────────── Medium ───────────────────────────────
@dataclass
class Medium:
    """A cultivation medium. `composition` is the human-readable ingredient map
    (component name → amount). Supply the 159-dimensional `vector` in the same
    component order used by the reference dataset. `params` holds optional
    non-composition condition variables (pH, temperature, time, growth phase)."""
    medium_id: str
    composition: dict[str, float] = field(default_factory=dict)  # ingredient → amount (semantic, human-readable)
    vector: Optional[list] = None       # required for the expression model; composition alone is not vectorized here
    params: dict[str, float] = field(default_factory=dict)      # pH, Temp, Time, Phase, ...


# ────────────────────────── Expression label ──────────────────────────
@dataclass
class ExpressionRecord:
    """One measurement: gene `gene_id` of genome `genome_id`, grown in medium
    `medium_id`, was expressed at `expression` (log-TPM)."""
    genome_id: str
    gene_id: str
    medium_id: str
    expression: float


# ─────────────────────────────── Dataset ──────────────────────────────
@dataclass
class Dataset:
    genomes: list[Genome] = field(default_factory=list)
    media: list[Medium] = field(default_factory=list)
    expression: list[ExpressionRecord] = field(default_factory=list)   # empty at inference

    # convenience lookups
    def genome(self, gid: str) -> Genome:
        return next(g for g in self.genomes if g.genome_id == gid)
    def medium(self, mid: str) -> Medium:
        return next(m for m in self.media if m.medium_id == mid)


def save_dataset(ds: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, values in [('genomes', ds.genomes), ('media', ds.media), ('expression', ds.expression)]:
        with open(os.path.join(out_dir, name + '.pkl'), 'wb') as handle:
            pickle.dump(values, handle)


def load_dataset(in_dir: str, with_labels: bool = True) -> Dataset:
    def load(name):
        with open(os.path.join(in_dir, name + '.pkl'), 'rb') as handle:
            return pickle.load(handle)
    expression = load('expression') if with_labels and os.path.exists(os.path.join(in_dir, 'expression.pkl')) else []
    return Dataset(genomes=load('genomes'), media=load('media'), expression=expression)
