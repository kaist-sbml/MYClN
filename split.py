"""Leakage-safe genus-level train/val/test split for a gene-expression dataset.

This is a single, self-contained utility (no package, no relative imports). It
builds a per-species taxonomy from the unified expression dataset plus the two
source metadata CSVs, then produces a deterministic, phylogeny- and
media-stratified split with hard leakage guards.

Two lookups underpin every split:

    species_key -> genus            (lineage unit for a genus-disjoint split)
    species_key -> genome_group     (dedup unit; identical / near-identical
                                     genomes must never straddle a split)

``genome_group`` collapses three concrete duplication hazards found in the data:
  1. the same assembly listed twice (e.g. ``GCF_000203835.1`` and
     ``GCF_000203835.1_M145`` are the same genome),
  2. the same ``GCF`` accession reused across the two data sources,
  3. the same organism (genus + species) present in both sources.

Public API
----------
``build_taxonomy(unified_data, dtu_meta_path=..., manual_meta_path=...)``
    -> ``dict[str, SpeciesInfo]``

``make_split(taxonomy, media_by_species)``
    -> ``Fold`` with ``.train_species`` / ``.val_species`` / ``.test_species``
    (each a tuple of species keys).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# ── external metadata paths (defaults) ───────────────────────────────────────
# GCF/NBC organism names are read from the two source metadata CSVs. These are
# the production paths; override them via the function arguments if the data
# lives elsewhere.
from config import DTU_META, MANUAL_META, ANTISMASH_DIR
# Some DTU strains have expression data + genomes but no row in the metadata CSV.
# Their taxonomy is recovered from the ORGANISM header of the per-strain antiSMASH
# GenBank file, so no genome with a resolvable lineage is silently dropped.
# Default recovery is disabled in config.py.

UNKNOWN = "UNKNOWN"

# The background genus is the bulk of the labelled data. It is always kept in
# training (never a test/val fold): holding it out would gut the model rather
# than measure transfer to a novel lineage.
BACKGROUND_GENUS = "Streptomyces"

# genus -> order within class Actinomycetia, used only to stratify the split so
# train/val/test each span the phylogenetic breadth of the class.
ORDER = {
    "Streptomyces": "Streptomycetales", "Kitasatospora": "Streptomycetales",
    "Embleya": "Streptomycetales", "Streptomycetaceae": "Streptomycetales",
    "Amycolatopsis": "Pseudonocardiales", "Lentzea": "Pseudonocardiales",
    "Saccharopolyspora": "Pseudonocardiales", "Actinosynnema": "Pseudonocardiales",
    "Actinomadura": "Streptosporangiales", "Spirillospora": "Streptosporangiales",
    "Nonomuraea": "Streptosporangiales",
    "Micromonospora": "Micromonosporales", "Kribbella": "Propionibacteriales",
    "Corynebacterium": "Corynebacteriales", "Nocardia": "Corynebacteriales",
    "Rhodococcus": "Corynebacteriales", "Mycolicibacterium": "Corynebacteriales",
}


@dataclass(frozen=True)
class SpeciesInfo:
    """Taxonomy for one labelled species (one key in the unified dataset)."""

    key: str            # species key as used in unified_expression_data.pkl
    source: str         # 'dtu_pep' | 'manual_curated'
    organism: str       # full organism string, e.g. 'Streptomyces canus'
    genus: str          # first token of organism; lineage unit for the split
    genome_group: str   # dedup unit: identical genomes share this id


@dataclass(frozen=True)
class Fold:
    """One split: everything needed to train and evaluate it."""

    test_genus: str
    train_species: tuple[str, ...]
    val_species: tuple[str, ...]
    test_species: tuple[str, ...]


# ── taxonomy construction ────────────────────────────────────────────────────
def _strip_variant_suffix(species_key: str) -> str:
    """Drop hand-added strain suffixes so genome accessions compare equal.

    ``GCF_000203835.1_M145`` and ``GCF_000203835.1`` are the same assembly.
    """
    return species_key.split("_M145")[0]


def _read_dtu_organism_by_strain(dtu_meta_path) -> dict[str, str]:
    """NBC strain id -> organism string (last column of the DTU metadata)."""
    organism_by_strain: dict[str, str] = {}
    with open(dtu_meta_path, encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) > 2 and row[1].startswith("NBC"):
                organism_by_strain[row[1]] = row[-1].strip()
    return organism_by_strain


def _read_manual_organism_by_accession(manual_meta_path) -> tuple[dict[str, str], dict[str, str]]:
    """GCF accession -> (organism string [col 5], genus [col 18]) from Manual.

    The explicit Genus column is a reliable fallback when the organism string
    is a bare ``Genus sp.`` placeholder.
    """
    organism_by_accession: dict[str, str] = {}
    genus_by_accession: dict[str, str] = {}
    with open(manual_meta_path, encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) > 4 and row[2].startswith("GCF"):
                organism_by_accession[row[2]] = row[4].strip()
                if len(row) > 17 and row[17].strip():
                    genus_by_accession[row[2]] = row[17].strip()
    return organism_by_accession, genus_by_accession


def _read_gbk_organism(strain: str, antismash_dir) -> str:
    """NBC strain -> organism string from its antiSMASH ``.gbk`` ORGANISM header.

    Used only as a fallback for strains missing from the DTU metadata CSV, so a
    real genome is never dropped from the split for lack of a metadata row.
    """
    if not antismash_dir:
        return ""
    gbk = Path(antismash_dir) / f"{strain}_antismash" / f"{strain}.gbk"
    if not gbk.exists():
        return ""
    with open(gbk) as handle:
        for line in handle:
            if line.strip().startswith("ORGANISM"):
                return line.split("ORGANISM", 1)[1].strip()
    return ""


def _organism_to_genus(organism: str) -> str:
    return organism.split()[0] if organism else UNKNOWN


# Species epithets that do not identify a real species; entries sharing only
# such a placeholder must NOT be merged into one genome_group.
_NON_SPECIFIC_EPITHETS = {"sp.", "sp", "spp.", "spp", "cf.", "aff."}


def _organism_to_species(organism: str) -> str | None:
    """Return 'Genus species' only for a resolved binomial name.

    Returns ``None`` when the name is missing or non-specific (e.g.
    'Streptomyces sp.'), so callers can skip organism-based deduplication for
    those entries.
    """
    tokens = organism.split()
    if len(tokens) < 2 or tokens[1].lower() in _NON_SPECIFIC_EPITHETS:
        return None
    return f"{tokens[0]} {tokens[1]}"


def build_taxonomy(
    unified_data: dict,
    dtu_meta_path=DTU_META,
    manual_meta_path=MANUAL_META,
    antismash_dir=ANTISMASH_DIR,
) -> dict[str, SpeciesInfo]:
    """Return ``species_key -> SpeciesInfo`` for every labelled species.

    ``unified_data`` is the loaded ``unified_expression_data.pkl``. Its
    ``species_metadata`` gives the authoritative species set and source; the
    raw source CSVs (``dtu_meta_path``, ``manual_meta_path``) supply organism
    names, and ``antismash_dir`` supplies the organism header for DTU strains
    absent from the CSV (so no real genome is dropped for lack of a metadata row).
    """
    metadata = unified_data["species_metadata"]
    dtu_organism = _read_dtu_organism_by_strain(dtu_meta_path)
    manual_organism, manual_genus = _read_manual_organism_by_accession(manual_meta_path)

    def organism_of(species_key: str) -> str:
        base = _strip_variant_suffix(species_key)
        if base.startswith("NBC"):
            org = dtu_organism.get(base, "")
            if not org:                      # absent from DTU CSV -> antiSMASH gbk header
                org = _read_gbk_organism(base, antismash_dir)
            return org
        return manual_organism.get(base, metadata[species_key].get("organism", ""))

    # First pass: resolve organism + genus for each species. When the organism
    # string is missing/non-specific, fall back to the explicit Manual Genus
    # column so the split never sees an UNKNOWN lineage for a real genome.
    resolved: dict[str, tuple[str, str, str]] = {}  # key -> (source, organism, genus)
    for species_key in metadata:
        source = metadata[species_key].get("source", UNKNOWN)
        organism = organism_of(species_key)
        genus = _organism_to_genus(organism)
        if genus == UNKNOWN:
            genus = manual_genus.get(_strip_variant_suffix(species_key), UNKNOWN)
        resolved[species_key] = (source, organism, genus)

    # Second pass: assign a genome_group that merges duplication hazards.
    # We union three keys onto a single canonical group id:
    #   - stripped genome accession (handles the _M145 duplicate + reuse),
    #   - 'Genus species' (handles the same organism across sources).
    # A tiny union-find keeps this order-independent and explicit.
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for species_key, (_, organism, _) in resolved.items():
        accession = _strip_variant_suffix(species_key)
        species_name = _organism_to_species(organism)
        # Link this species to its accession-token (collapses the _M145 dup and
        # any GCF reused across sources). Only link by organism when the name is
        # a resolved binomial -- never for 'Genus sp.', which would wrongly fuse
        # dozens of distinct genomes into one group.
        union(species_key, f"acc::{accession}")
        if species_name is not None:
            union(species_key, f"org::{species_name}")

    taxonomy: dict[str, SpeciesInfo] = {}
    for species_key, (source, organism, genus) in resolved.items():
        taxonomy[species_key] = SpeciesInfo(
            key=species_key,
            source=source,
            organism=organism or UNKNOWN,
            genus=genus,
            genome_group=find(species_key),
        )
    return taxonomy


# ── leakage guard ────────────────────────────────────────────────────────────
def _assert_no_leakage(fold: Fold, taxonomy: dict[str, SpeciesInfo]) -> None:
    """Fail loudly if any leakage guard is violated."""
    train, val, test = set(fold.train_species), set(fold.val_species), set(fold.test_species)

    # 1. the three sets are disjoint at the species level
    assert train.isdisjoint(val), "train/val overlap"
    assert train.isdisjoint(test), "train/test overlap"
    assert val.isdisjoint(test), "val/test overlap"

    # 2. genus-disjoint: test/val genera never bleed into train
    genus = lambda keys: {taxonomy[k].genus for k in keys}
    assert genus(test).isdisjoint(genus(train)), "test genus present in train"
    assert genus(val).isdisjoint(genus(train)), "val genus present in train"
    assert genus(test).isdisjoint(genus(val)), "test genus present in val"

    # 3. no genome_group straddles the split
    group = lambda keys: {taxonomy[k].genome_group for k in keys}
    assert group(train).isdisjoint(group(test)), "genome_group straddles train/test"
    assert group(train).isdisjoint(group(val)), "genome_group straddles train/val"
    assert group(val).isdisjoint(group(test)), "genome_group straddles val/test"


# ── the split ────────────────────────────────────────────────────────────────
def make_split(taxonomy, media_by_species):
    """Single train/val/test split, stratified by taxonomic order and balanced
    for media coverage. Deterministic.

    Rule (per order): the genus with the most genome-groups -> TEST (stable
    novel-lineage estimate, every order incl. distant represented); among the
    rest, the genus with the highest overlap with the shared core media -> VAL
    (so validation-based model selection sees train-covered conditions); the
    remainder + the background genus -> TRAIN. Returns a Fold; leakage guards
    (genus- and genome_group-disjoint) are asserted.
    """
    resolved = {k: i for k, i in taxonomy.items()
                if i.genus != UNKNOWN and k in media_by_species}
    groups, media = {}, {}
    for k, info in resolved.items():
        groups.setdefault(info.genus, set()).add(info.genome_group)
        media.setdefault(info.genus, set()).update(media_by_species[k])
    genera = [g for g in groups if g != BACKGROUND_GENUS]

    core = media.get(BACKGROUND_GENUS, set())
    overlap = {g: len(media[g] & core) / max(len(media[g]), 1) for g in genera}
    by_order = {}
    for g in genera:
        by_order.setdefault(ORDER.get(g, "Other"), []).append(g)

    test_genera, val_genera = [], []
    for od in sorted(by_order):
        ranked = sorted(by_order[od], key=lambda g: (-len(groups[g]), g))
        test_genera.append(ranked[0])
        rest = ranked[1:]
        if rest:
            val_genera.append(max(rest, key=lambda g: (overlap[g], len(groups[g]), g)))
    test_g, val_g = set(test_genera), set(val_genera)

    def species_in(pred):
        return tuple(sorted(k for k, i in resolved.items() if pred(i.genus)))
    fold = Fold(
        test_genus="__genus_split__",
        train_species=species_in(lambda g: g not in test_g and g not in val_g),
        val_species=species_in(lambda g: g in val_g),
        test_species=species_in(lambda g: g in test_g),
    )
    _assert_no_leakage(fold, taxonomy)
    return fold
