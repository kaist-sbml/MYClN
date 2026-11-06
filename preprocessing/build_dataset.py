#!/usr/bin/env python
"""Raw sources  ->  standard-schema Dataset (genomes.pkl / media.pkl / expression.pkl).

Distilled from the reference notebook `build_from_raw.ipynb`. It parses the two raw
Actinomycete collections, unifies their media/condition descriptions into one fixed
media vocabulary, attaches per-gene sequences and per-condition expression, and emits
objects conforming to `schema.py` via `schema.save_dataset`.

Producer pipeline (this file):
    raw genomes + RNA-seq + media (DTU_Pep + Manual_curated)
        -> internal table  {gene_wise_data, species_metadata, component_list}
        -> standard-schema Dataset (Genome / Gene / Medium / ExpressionRecord)

Required raw inputs (external — NOT bundled; this script is not runnable without them):
  Shared lists / ontology
    <RAW_DATA_DIR>/Manual_curated_data/Whole_list/*.txt   (EC/DUF/Pfam/BGC/supplement vocab)
    <TOOLS_DIR>/Pfam/pfam2go                          (Pfam -> GO map)
  Media
    <RAW_DATA_DIR>/DTU_Pep_data/Media.csv                 (CSV media compositions)
    <RAW_DATA_DIR>/Manual_curated_data/Metadata/Media.json(JSON media compositions)
  DTU_Pep collection
    antismash_results/<sp>_antismash/<sp>.gbk                (GenBank: CDS aa/dna, BGC)
    log_tpm/<sp>/<sp>_log_gm.csv                             (per-condition log-TPM)
    Whole_genome_inputs/operons_t0p5.tsv + promoters FASTA   (operon layout + promoters)
    Pfam_analysis_results/, pep_ecpredict2/                  (optional genome features)
  Manual_curated collection
    Genome/<sp>.gbk, Genome/operons_t0p5.tsv + promoters FASTA
    TPM/<geo>_<genome>_log2TPM.tsv                           (per-sample log2-TPM)
    Metadata/Actinomycetes_metadata_20260106.csv            (sample -> media/condition)
    Pfam_results/, deepec.filtered_by_ec_count.tsv          (optional genome features)

External tools that produced the raw inputs (also not bundled):
  - a gene caller (CDS calling)         -> GenBank CDS + cds_meta.json/cds_emb.npy
  - an operon predictor                 -> operons_t0p5.tsv + promoters_*.fasta
  - ESM-2 (esm2_t12_35M_UR50D, 480-d)   -> per-gene protein embeddings (see esm_embeddings.py)
  - antiSMASH + HMMER/Pfam + DeepEC     -> BGC / Pfam / EC features (OPTIONAL: the deployed
                                           model is BGC-free; these do not enter schema.py)

Only sequences (aa/dna/promoter), expression (log-TPM) and media compose the standard
schema. Pfam / EC / BGC are optional genome features kept in the internal table for
analyses and do NOT map into schema.py.

Output:
  outputs/dataset/{genomes.pkl, media.pkl, expression.pkl}   (schema.save_dataset)
  data/unified_expression_data.pkl                               (internal table; downstream)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as release_config
import os
import re
import sys
import json
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# schema.py lives one directory up (repository root/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import Genome, Gene, Medium, ExpressionRecord, Dataset, save_dataset


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Config — paths rooted in config.py RAW_DATA_DIR / TOOLS_DIR
# ══════════════════════════════════════════════════════════════════════════════
SHARED_LIST_DIR = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Whole_list')
PFAM2GO_PATH    = os.path.join(release_config.TOOLS_DIR, 'Pfam/pfam2go')

# media compositions (two sources, unified into one vocabulary)
DTU_MEDIA_CSV   = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/Media.csv')
MANUAL_MEDIA_JSON = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Metadata/Media.json')

# DTU_Pep collection
DTU_GENOME_DIR   = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/antismash_results')
DTU_TPM_DIR      = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/log_tpm')
DTU_PFAM_DIR     = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/Pfam_analysis_results')
DTU_DEEPEC_DIR   = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/pep_ecpredict2/pep_ecpredict')
DTU_OPERON_TSV   = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/Whole_genome_inputs/operons_t0p5.tsv')
DTU_PROMOTER_FASTA = os.path.join(release_config.RAW_DATA_DIR, 'DTU_Pep_data/Whole_genome_inputs/promoters_up250_down50_t0p5_2.fasta')

# Manual_curated collection
MANUAL_GENOME_DIR = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Genome')
MANUAL_TPM_DIR    = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/TPM')
MANUAL_PFAM_DIR   = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Pfam_results')
MANUAL_DEEPEC_TSV = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/deepec.filtered_by_ec_count.tsv')
MANUAL_METADATA_CSV = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Metadata/Actinomycetes_metadata_20260106.csv')
MANUAL_OPERON_TSV = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Genome/operons_t0p5.tsv')
MANUAL_PROMOTER_FASTA = os.path.join(release_config.RAW_DATA_DIR, 'Manual_curated_data/Genome/promoters_up250_down50_t0p5_2.fasta')

# outputs
_HERE       = os.path.dirname(os.path.abspath(__file__))
_RELEASE    = os.path.dirname(_HERE)
_FINAL      = os.path.dirname(_RELEASE)
DATASET_DIR = release_config.DATASET_DIR                               # standard schema
INTERNAL_PKL = release_config.path("BUILD_INTERNAL_PKL", str(release_config.ROOT / "outputs/unified_expression_data.pkl"))    # internal table

# Pfam -> GO targets: transcription / regulation ontology terms
TARGET_GOS = {"GO:0006355", "GO:0003700", "GO:0006352", "GO:0000160", "GO:0007165"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Shared list / feature helpers
#     (Pfam / EC / BGC are OPTIONAL genome features; not part of schema.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_pfam2go(pfam2go_path: str, target_gos: Set[str]) -> dict:
    go2pfam: Dict[str, set] = defaultdict(set)
    with open(pfam2go_path) as f:
        for line in f:
            if line.startswith("!"):
                continue
            for go in target_gos:
                if go in line:
                    pfam = line.split()[0].replace("Pfam:", "")
                    go2pfam[go].add(pfam)
    pfam2go = {}
    for go in sorted(target_gos):
        for pfam in go2pfam[go]:
            pfam2go.setdefault(pfam, []).append(go)
    return pfam2go


def hmmer_parser(
    domtblout_file: str,
    pfam2go: dict,
    whole_duf_list: List[str],
    whole_pfam_list: List[str],
    IEVALUE_CUTOFF: float = 1e-3,
    COV_CUTOFF: float = 0.7,
) -> Tuple[List[float], List[float]]:
    """OPTIONAL: normalized regulatory-DUF / regulatory-Pfam count vectors from an
    HMMER domtblout. Not used by the deployed model."""
    gene_to_domains: dict = {}
    with open(domtblout_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.strip().split()
            desc = " ".join(cols[22:])
            pfam_id = cols[1].split(".")[0]
            query = cols[3]
            i_evalue = float(cols[12])
            coverage = float(cols[21])
            ali_from = int(cols[17])
            ali_to = int(cols[18])

            if not (i_evalue <= IEVALUE_CUTOFF and coverage >= COV_CUTOFF):
                continue
            if not (pfam_id in pfam2go or "DUF" in desc):
                continue

            gene_to_domains.setdefault(query, []).append({
                "pfam": pfam_id, "ali_from": ali_from, "ali_to": ali_to,
                "i_evalue": i_evalue, "coverage": coverage, "GO": pfam2go.get(pfam_id),
            })

    duf_cnt = {d: 0 for d in whole_duf_list}
    pfam_cnt = {p: 0 for p in whole_pfam_list}
    for gene in gene_to_domains:
        for domain in gene_to_domains[gene]:
            if domain["GO"] is None:
                if domain["pfam"] in duf_cnt:
                    duf_cnt[domain["pfam"]] += 1
            else:
                if domain["pfam"] in pfam_cnt:
                    pfam_cnt[domain["pfam"]] += 1

    duf_vec = [duf_cnt[d] for d in whole_duf_list]
    total_d = sum(duf_vec)
    norm_duf = [c / total_d if total_d > 0 else 0.0 for c in duf_vec]
    pfam_vec = [pfam_cnt[p] for p in whole_pfam_list]
    total_p = sum(pfam_vec)
    norm_pfam = [c / total_p if total_p > 0 else 0.0 for c in pfam_vec]
    return norm_duf, norm_pfam


def parse_bgc_information(
    records,
    bgc_local_regulator_list: List[str],
    bgc_tfbs_list: List[str],
    whole_bgc_type_list: List[str],
):
    """OPTIONAL: parse antiSMASH region annotations into per-BGC type / TFBS /
    local-regulator one-hots + a locus_tag -> BGC map. The deployed model is BGC-free;
    this is retained only for optional analyses and is not exported to schema.py."""
    boundaries: dict = {}
    bgc_id = 0
    for record in records:
        boundaries[record.id] = []
        for feature in record.features:
            if feature.type == "region":
                bgc_product = feature.qualifiers.get("product", [""])
                boundaries[record.id].append((
                    (int(feature.location.start), int(feature.location.end)),
                    bgc_product, bgc_id,
                ))
                bgc_id += 1

    tfbs: dict = {}
    local_regulator: dict = {}
    bgc_type: dict = {}
    locus_tag_to_bgc: dict = {}

    for record in records:
        for feature in record.features:
            fp = (int(feature.location.start), int(feature.location.end))
            for (bs, be), bp, bid in boundaries[record.id]:
                if bs <= fp[0] and fp[1] <= be:
                    tfbs.setdefault(bid, [])
                    local_regulator.setdefault(bid, [])
                    bgc_type[bid] = bp

                    if feature.type == "CDS":
                        lt = feature.qualifiers.get("locus_tag", [""])[0]
                        locus_tag_to_bgc[lt] = bid
                        if "gene_kind" in feature.qualifiers:
                            if feature.qualifiers["gene_kind"][0] == "regulatory":
                                gf = feature.qualifiers.get("gene_functions", [""])
                                gf = [x for x in gf if "regulatory" in x.lower()]
                                if gf:
                                    local_regulator[bid].append(gf[0].split(": ")[-1])

                    if feature.type == "misc_feature":
                        note = feature.qualifiers.get("note", [""])[0]
                        tf_m = re.search(r"\bto\s+([^,]+)", note)
                        cf_m = re.search(r"confidence:\s*([^,]+)", note)
                        tf_name = tf_m.group(1).strip() if tf_m else None
                        conf = cf_m.group(1).strip() if cf_m else None
                        if conf in ("high", "medium"):
                            tfbs[bid].append({"tf_name": tf_name, "confidence": conf})

    bgc_feature = {}
    for bid in bgc_type:
        tf_set = {x["tf_name"] for x in tfbs.get(bid, []) if x["tf_name"]}
        bgc_feature[bid] = {
            "type": [1 if t in bgc_type[bid] else 0 for t in whole_bgc_type_list],
            "tfbs": [1 if t in tf_set else 0 for t in bgc_tfbs_list],
            "local_regulator": [
                1 if lr in local_regulator.get(bid, []) else 0
                for lr in bgc_local_regulator_list
            ],
        }
    return locus_tag_to_bgc, bgc_feature


def load_promoter_seqs(promoter_fasta_path: str) -> dict:
    """operon_id (genome|contig|operon) -> promoter sequence."""
    seq_dict = {}
    with open(promoter_fasta_path) as f:
        header = None
        seq_lines = []
        operon_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seq_dict[operon_id] = "".join(seq_lines)
                header = line[1:]
                operon_id = "|".join(header.split("|")[:3])
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            seq_dict[operon_id] = "".join(seq_lines)
    return seq_dict


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Media vectorizer — one fixed vocabulary across both sources
#     (this IS the MediaVectorizer / media_vocab referenced by schema.py Medium)
# ══════════════════════════════════════════════════════════════════════════════
class MediaVectorizer:
    """Unifies CSV media (DTU_Pep) + JSON media (Manual_curated) into a single sorted
    component list (the media vocabulary) and maps any media name -> fixed-dim vector.
    `component_list` is the media_vocab; `encode(name)` / `dict_to_vec(comp)` build the
    model-ready `Medium.vector`."""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
        supplement_list: Optional[List[str]] = None,
    ):
        # 1) CSV-based media (DTU_Pep)
        self.csv_media: Dict[str, Dict[str, float]] = {}
        csv_components: Set[str] = set()
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path).fillna(0)
            comp_cols = [c for c in df.columns if c != "Media_type"]
            csv_components.update(comp_cols)
            for _, row in df.iterrows():
                self.csv_media[row["Media_type"]] = {c: float(row[c]) for c in comp_cols}

        # 2) JSON-based media (Manual_curated)
        self.json_media: Dict[str, Dict[str, float]] = {}
        json_components: Set[str] = set()
        if json_path and os.path.exists(json_path):
            with open(json_path) as f:
                media_data = json.load(f)
            for media_name, val in media_data.items():
                if isinstance(val, dict):
                    self.json_media[media_name] = {
                        k: float(v) if isinstance(v, (int, float)) else 1.0
                        for k, v in val.items()
                    }
                    json_components.update(val.keys())
                elif isinstance(val, str):
                    self.json_media[media_name] = {val: 1.0}   # plain string -> binary 1
                    json_components.add(val)

        # 3) supplement vocabulary
        sup_set = set(supplement_list) if supplement_list else set()

        # 4) fixed component list = the media vocabulary
        all_components = csv_components | json_components | sup_set
        self.component_list = sorted(all_components)
        self.comp_to_idx = {c: i for i, c in enumerate(self.component_list)}
        self.component_dim = len(self.component_list)

        # 5) cache: media name -> vector
        self._cache: Dict[str, np.ndarray] = {}
        for name, comp_dict in {**self.csv_media, **self.json_media}.items():
            self._cache[name] = self.dict_to_vec(comp_dict)

    def dict_to_vec(self, comp_dict: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(self.component_dim, dtype=np.float32)
        for comp, val in comp_dict.items():
            if comp in self.comp_to_idx:
                vec[self.comp_to_idx[comp]] = val
        return vec

    def encode(self, media_name: str) -> np.ndarray:
        if media_name in self._cache:
            return self._cache[media_name].copy()
        return np.zeros(self.component_dim, dtype=np.float32)   # unknown -> zero

    def vec_to_composition(self, vec) -> Dict[str, float]:
        """Fixed-dim vector -> human-readable {component: amount} (nonzero entries)."""
        vec = np.asarray(vec, dtype=np.float32)
        return {self.component_list[i]: float(v) for i, v in enumerate(vec) if v != 0}

    def available_media(self) -> List[str]:
        return sorted(self._cache.keys())


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Per-source loaders  ->  internal gene_wise_data + species_metadata
#     gene_wise_data[genome][gene_id] = {gene_info, conditions[], source}
# ══════════════════════════════════════════════════════════════════════════════

# ── DTU_Pep: TPM column substring -> canonical media name ─────────────────────
DTU_MEDIA_KEY = {
    "DNPM": "DNPM", "ISP2": "ISP2", "MA": "MA", "SoyM": "soyM",
    "TSB": "TSB50", "gluc": "SMM-glucose", "gly": "SMM-glycerol", "malt": "SMM-maltose",
}


def build_gene_wise_data_dtu(
    species_list: List[str],
    genome_dir: str,
    tpm_dir: str,
    pfam_results_dir: str,
    deepec_results_dir: str,
    operon_df: pd.DataFrame,
    media_vectorizer: MediaVectorizer,
    pfam2go: dict,
    whole_duf_list: List[str],
    whole_pfam_list: List[str],
    whole_ec_list: List[str],
    bgc_local_regulator_list: List[str],
    bgc_tfbs_list: List[str],
    whole_bgc_type_list: List[str],
) -> Tuple[dict, dict]:
    species_metadata = {}
    gene_wise_data = {}

    for species in tqdm(species_list, desc="[DTU_Pep] Processing"):
        try:
            species_metadata[species] = {"source": "dtu_pep"}

            # ── OPTIONAL genome features (Pfam / EC) ─────────
            pfam_file = os.path.join(pfam_results_dir, species, "pfam_domains.tbl")
            if os.path.exists(pfam_file):
                species_metadata[species]["regulatory_pfam"] = hmmer_parser(
                    pfam_file, pfam2go, whole_duf_list, whole_pfam_list)
            else:
                species_metadata[species]["regulatory_pfam"] = (
                    [0.0] * len(whole_duf_list), [0.0] * len(whole_pfam_list))

            ec_path = os.path.join(deepec_results_dir, species, species, "DeepECv2_result.txt")
            ec_set: Set[str] = set()
            if os.path.exists(ec_path):
                ec_df = pd.read_csv(ec_path, sep="\t")
                for _, row in ec_df.iterrows():
                    pred = row.get("prediction")
                    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                        continue
                    if len(str(pred).split(";")) < 3:
                        ec_set.update(x.replace("EC:", "") for x in str(pred).split(";"))
            species_metadata[species]["ec_list"] = [1 if ec in ec_set else 0 for ec in whole_ec_list]

            # ── GenBank: per-CDS aa/dna sequences + (optional) BGC ──
            genome_file = os.path.join(genome_dir, f"{species}_antismash", f"{species}.gbk")
            if not os.path.exists(genome_file):
                print(f"  [SKIP] GBK not found: {genome_file}")
                continue
            records = list(SeqIO.parse(genome_file, "genbank"))

            locus_tag_to_dna, locus_tag_to_aa = {}, {}
            for record in records:
                for feature in record.features:
                    if feature.type != "CDS":
                        continue
                    lt = feature.qualifiers.get("locus_tag", [None])[0]
                    if lt is None:
                        continue
                    locus_tag_to_dna[lt] = str(feature.location.extract(record.seq))
                    locus_tag_to_aa[lt] = feature.qualifiers.get("translation", [None])[0]

            locus_tag_to_bgc, bgc_info = parse_bgc_information(
                records, bgc_local_regulator_list, bgc_tfbs_list, whole_bgc_type_list)
            species_metadata[species]["bgc_information"] = bgc_info   # OPTIONAL

            # ── operon -> per-gene info (promoter + sequences) ──
            sp_operon = operon_df[operon_df["genome"] == species]
            gene_id_to_operon: dict = {}
            for _, op_row in sp_operon.iterrows():
                oid = op_row["operon_id"]
                for idx, gene in enumerate(str(op_row["genes"]).split(",")):
                    gene_id_to_operon[gene] = {
                        "operon_id": oid,
                        "gene_index_in_operon": idx + 1,
                        "promoter_sequence": op_row.get("sequence"),
                        "gene_sequence_dna": locus_tag_to_dna.get(gene),
                        "gene_sequence_aa": locus_tag_to_aa.get(gene),
                        "bgc_id": locus_tag_to_bgc.get(gene),   # OPTIONAL
                    }

            # ── per-condition expression (replicate-averaged log-TPM) ──
            tpm_file = os.path.join(tpm_dir, species, f"{species}_log_gm.csv")
            if not os.path.exists(tpm_file):
                print(f"  [SKIP] TPM not found: {tpm_file}")
                continue
            tpm_df = pd.read_csv(tpm_file, index_col=0)

            col_grouping: Dict[str, list] = {}
            for col in tpm_df.columns:
                for key in DTU_MEDIA_KEY:
                    if key in col:
                        col_grouping.setdefault(key, []).append(col)
                        break
            expression_df = pd.DataFrame(index=tpm_df.index)
            for key, cols in col_grouping.items():
                expression_df[DTU_MEDIA_KEY[key]] = tpm_df[cols].mean(axis=1)

            # ── assemble ──
            gene_wise_data[species] = {}
            for gene_id, gene_info in gene_id_to_operon.items():
                conditions = []
                for condition in expression_df.columns:
                    media_vec = media_vectorizer.encode(condition)
                    expr = expression_df[condition].get(gene_id, None)
                    if isinstance(expr, pd.Series):
                        expr = expr.mean()
                    conditions.append({
                        "media_name": condition, "media_vector": media_vec, "expression": expr,
                    })
                gene_wise_data[species][gene_id] = {
                    "gene_info": gene_info, "conditions": conditions, "source": "dtu_pep",
                }
        except Exception as e:
            print(f"  [ERROR] {species}: {e}")

    return gene_wise_data, species_metadata


# ── Manual_curated: supplement unit conversion (-> the media vocabulary units) ──
SUPPLEMENT_CONVERSION = {
    ("CaCO₃", "mM"):       (("CaCO3", "g/L"),         lambda v, m: (v * m) / 1000, 100.09),
    ("CaCl₂", "mM"):       (("CaCl2", "mM"),          lambda v, *_: v,             None),
    ("CaCl2·2H2O", "g/L"): (("CaCl2", "mM"),          lambda v, m: (v * 1000) / m, 147.01),
    ("CoCl₂·5H₂O", "mM"): (("CoCl2·6H2O", "g/L"),    lambda v, m: (v * m) / 1000, 237.93),
    ("CuSO₄·5H₂O", "mM"): (("CuSO4·5H2O", "g/L"),    lambda v, m: (v * m) / 1000, 249.68),
    ("FeSO₄·7H₂O", "mM"): (("FeSO4·7H2O", "g/L"),     lambda v, m: (v * m) / 1000, 278.01),
    ("H₃BO₃", "mM"):      (("H3BO3", "mM"),           lambda v, *_: v,             None),
    ("H3BO3", "g/L"):      (("H3BO3", "mM"),           lambda v, m: (v * 1000) / m, 61.83),
    ("MgCl₂·6H₂O", "mM"): (("MgCl2·6H2O", "mM"),     lambda v, *_: v,             None),
    ("MgSO₄", "mM"):      (("MgSO4", "mM"),           lambda v, *_: v,             None),
    ("MnSO₄·H₂O", "mM"):  (("MnSO4·4H2O", "mM"),     lambda v, *_: v,             None),
    ("MnSO4·H2O", "g/L"):  (("MnSO4·4H2O", "mM"),     lambda v, m: (v * 1000) / m, 197.91),
    ("MnSo4·4H2O", "g/L"): (("MnSO4·4H2O", "mM"),     lambda v, m: (v * 1000) / m, 197.91),
    ("ZnSO₄·7H₂O", "mM"): (("ZnSO4·7H2O", "g/L"),    lambda v, m: (v * m) / 1000, 287.54),
    ("HCl", "g/L"):        (("HCl", "M"),              lambda v, m: v / m,          36.46),
}

EXCLUDED_SAMPLES = {
    "GSM3133289", "GSM7327792", "GSM1038627",
    "GSM1362152", "GSM1362153", "GSM1362154", "GSM1362155",
}

SUPPLE_PATTERN = re.compile(r"\s*([^();]+)\s*\(\s*([\d.]+)\s*([^\)]+)\s*\)")


def _parse_supplements(supplements_str: str) -> List[Tuple[str, float, str]]:
    """Supplement string -> unit-converted (name, value, unit) list."""
    if pd.isna(supplements_str) or not supplements_str.strip():
        return []
    matches = SUPPLE_PATTERN.findall(str(supplements_str))
    result = []
    for name_raw, value_raw, unit_raw in matches:
        name = name_raw.strip()
        value = float(value_raw)
        unit = unit_raw.strip()
        key = (name, unit)
        if key in SUPPLEMENT_CONVERSION:
            (conv_name, conv_unit), conv_fn, molar_mass = SUPPLEMENT_CONVERSION[key]
            value = conv_fn(value, molar_mass) if molar_mass is not None else conv_fn(value)
            name, unit = conv_name, conv_unit
        result.append((name, value, unit))
    return result


def build_gene_wise_data_manual(
    species_list: List[str],
    genome_dir: str,
    tpm_dir: str,
    pfam_results_dir: str,
    deepec_tsv_path: str,
    operon_df: pd.DataFrame,
    metadata_csv_path: str,
    media_vectorizer: MediaVectorizer,
    media_json_path: str,
    pfam2go: dict,
    whole_duf_list: List[str],
    whole_pfam_list: List[str],
    whole_ec_list: List[str],
    bgc_local_regulator_list: List[str],
    bgc_tfbs_list: List[str],
    whole_bgc_type_list: List[str],
    excluded_samples: Optional[Set[str]] = None,
) -> Tuple[dict, dict]:
    if excluded_samples is None:
        excluded_samples = EXCLUDED_SAMPLES

    metadata_df = pd.read_csv(metadata_csv_path)
    with open(media_json_path) as f:
        media_data = json.load(f)
    deepec_all = pd.read_csv(deepec_tsv_path, sep="\t") if os.path.exists(deepec_tsv_path) else None

    # ── Step 1: per-genome genome/operon parsing (+ optional Pfam/EC/BGC) ──
    species_metadata: dict = {}
    gene_sequence_data: dict = {}

    for species in tqdm(species_list, desc="[Manual] Genome parsing"):
        try:
            species_metadata[species] = {"source": "manual_curated"}

            # OPTIONAL Pfam
            pfam_file = os.path.join(pfam_results_dir, f"{species}.tbl")
            if os.path.exists(pfam_file):
                species_metadata[species]["regulatory_pfam"] = hmmer_parser(
                    pfam_file, pfam2go, whole_duf_list, whole_pfam_list)
            else:
                species_metadata[species]["regulatory_pfam"] = (
                    [0.0] * len(whole_duf_list), [0.0] * len(whole_pfam_list))

            # OPTIONAL EC (single TSV keyed by genome)
            ec_set: Set[str] = set()
            if deepec_all is not None:
                sp_ec = deepec_all[deepec_all["genome"] == species]
                for _, row in sp_ec.iterrows():
                    ec_str = row.get("ec_list")
                    if ec_str is None or (isinstance(ec_str, float) and np.isnan(ec_str)):
                        continue
                    ec_set.update(str(ec_str).split(";"))
            species_metadata[species]["ec_list"] = [1 if ec in ec_set else 0 for ec in whole_ec_list]

            # GenBank: per-CDS aa/dna + (optional) BGC
            genome_file = os.path.join(genome_dir, f"{species}.gbk")
            if not os.path.exists(genome_file):
                print(f"  [SKIP] GBK not found: {genome_file}")
                continue
            records = list(SeqIO.parse(genome_file, "genbank"))

            locus_tag_to_dna, locus_tag_to_aa = {}, {}
            for record in records:
                for feature in record.features:
                    if feature.type != "CDS":
                        continue
                    lt = feature.qualifiers.get("locus_tag", [None])[0]
                    if lt is None:
                        continue
                    locus_tag_to_dna[lt] = str(feature.location.extract(record.seq))
                    locus_tag_to_aa[lt] = feature.qualifiers.get("translation", [None])[0]

            locus_tag_to_bgc, bgc_info = parse_bgc_information(
                records, bgc_local_regulator_list, bgc_tfbs_list, whole_bgc_type_list)
            species_metadata[species]["bgc_information"] = bgc_info   # OPTIONAL

            sp_operon = operon_df[operon_df["genome"] == species]
            gene_id_to_operon: dict = {}
            for _, op_row in sp_operon.iterrows():
                oid = op_row["operon_id"]
                for idx, gene in enumerate(str(op_row["genes"]).split(",")):
                    gene_id_to_operon[gene] = {
                        "operon_id": oid,
                        "gene_index_in_operon": idx + 1,
                        "promoter_sequence": op_row.get("sequence"),
                        "gene_sequence_dna": locus_tag_to_dna.get(gene),
                        "gene_sequence_aa": locus_tag_to_aa.get(gene),
                        "bgc_id": locus_tag_to_bgc.get(gene),   # OPTIONAL
                    }
            gene_sequence_data[species] = gene_id_to_operon
        except Exception as e:
            print(f"  [ERROR] genome parse {species}: {e}")

    # ── Step 2: metadata filtering + condition grouping ──
    #   Drop contig-level genomes, drop mutants (keep only SCP1-;SCP2-), drop excluded GEO.
    key_to_samples: Dict[str, List[str]] = {}
    sample_meta_tmp: Dict[str, dict] = {}

    for _, row in metadata_df.iterrows():
        if row.get("Genome level") == "Contig":
            continue
        mutation = row.get("Mutation")
        if pd.notna(mutation) and mutation != "SCP1-; SCP2-":
            continue

        genome_acc = str(row["Genome Accession"])
        if row.get("Mutation") == "SCP1-; SCP2-":
            genome_acc += "_M145"

        geo_acc = str(row["GEO Accession"])
        if geo_acc in excluded_samples:
            continue

        media_name = str(row.get("Media", ""))
        supplements = _parse_supplements(str(row.get("Supplements", "")))
        pH, Temp, RPM = row.get("pH"), row.get("Temp"), row.get("RPM")
        Time, Phase = row.get("Time(h)"), row.get("Phase")
        V_media, V_reactor = row.get("Media volume"), row.get("Reactor")

        # unique key: group same-condition replicates
        sup_key = tuple(sorted((n, round(v, 6), u) for n, v, u in supplements))
        unique_key = "__".join(map(str, [
            genome_acc, media_name, sup_key, pH, Temp, RPM, Time, Phase, V_media, V_reactor]))
        key_to_samples.setdefault(unique_key, []).append(geo_acc)

        sample_meta_tmp[geo_acc] = {
            "genome_accession": genome_acc, "media": media_name, "supplements": supplements,
            "pH": pH, "Temp": Temp, "RPM": RPM, "Time": Time, "Phase": Phase,
            "V_media": V_media, "V_reactor": V_reactor, "unique_key": unique_key,
        }

    # ── Step 3: replicate-averaged TPM + media vector per condition ──
    sample_metadata_grouped: Dict[str, list] = {}
    for unique_key, geo_accs in key_to_samples.items():
        genome_acc = sample_meta_tmp[geo_accs[0]]["genome_accession"]

        tpm_frames = []
        for geo_acc in geo_accs:
            tpm_file = os.path.join(tpm_dir, f"{geo_acc}_{genome_acc}_log2TPM.tsv")
            if os.path.exists(tpm_file):
                df = pd.read_csv(tpm_file, sep="\t", index_col=0)
                df = df.rename(columns={"log2_TPM": geo_acc})
                tpm_frames.append(df[[geo_acc]])
            else:
                print(f"  [SKIP] TPM not found: {tpm_file}")
        if not tpm_frames:
            continue
        averaged_tpm = pd.concat(tpm_frames, axis=1).mean(axis=1)

        # media vector = base media (JSON) + supplements, via the shared vocabulary
        meta_info = sample_meta_tmp[geo_accs[0]]
        base_media_comp = media_data.get(meta_info["media"], {})
        comp_dict: Dict[str, float] = {}
        if isinstance(base_media_comp, dict):
            for comp, val in base_media_comp.items():
                comp_dict[comp] = comp_dict.get(comp, 0) + (
                    float(val) if isinstance(val, (int, float)) else 1.0)
        for sup_name, sup_val, _sup_unit in meta_info["supplements"]:
            comp_dict[sup_name] = comp_dict.get(sup_name, 0) + sup_val
        media_vec = media_vectorizer.dict_to_vec(comp_dict)

        condition_metadata = {
            "pH": meta_info["pH"], "Temp": meta_info["Temp"], "RPM": meta_info["RPM"],
            "Time": meta_info["Time"], "Phase": meta_info["Phase"],
            "V_media": meta_info["V_media"], "V_reactor": meta_info["V_reactor"],
        }
        sample_metadata_grouped.setdefault(genome_acc, []).append({
            "metadata": condition_metadata, "media_vector": media_vec,
            "media_name": meta_info["media"], "averaged_tpm": averaged_tpm.to_dict(),
        })

    # ── Step 4: assemble gene_wise_data ──
    final_species = set(gene_sequence_data.keys()) & set(sample_metadata_grouped.keys())
    gene_wise_data: dict = {}
    for species in final_species:
        gene_wise_data[species] = {}
        for gene_id, gene_info in gene_sequence_data[species].items():
            conditions = []
            for cond_entry in sample_metadata_grouped[species]:
                expr = cond_entry["averaged_tpm"].get(gene_id, None)
                if isinstance(expr, pd.Series):
                    expr = expr.mean()
                conditions.append({
                    "media_name": cond_entry["media_name"],
                    "media_vector": cond_entry["media_vector"],
                    "expression": expr,
                    "condition_metadata": cond_entry["metadata"],
                })
            gene_wise_data[species][gene_id] = {
                "gene_info": gene_info, "conditions": conditions, "source": "manual_curated",
            }

    print(f"[Manual] Parsed {len(gene_sequence_data)} genomes, "
          f"{len(sample_metadata_grouped)} with TPM, {len(gene_wise_data)} final species")
    return gene_wise_data, species_metadata


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Internal table  ->  standard-schema Dataset
# ══════════════════════════════════════════════════════════════════════════════
_PHASE_CODE = {"early": 1.0, "transition": 2.0, "late": 3.0, "stationary": 4.0}


def _numeric_params(cond_meta: dict) -> Dict[str, float]:
    """condition_metadata -> Medium.params (numeric only; Phase mapped to a code)."""
    out: Dict[str, float] = {}
    for k in ("pH", "Temp", "RPM", "Time"):
        v = cond_meta.get(k)
        try:
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                out[k] = float(v)
        except (TypeError, ValueError):
            pass
    phase = cond_meta.get("Phase")
    if isinstance(phase, str):
        for tag, code in _PHASE_CODE.items():
            if tag in phase.lower():
                out["Phase"] = code
                break
    return out


def to_standard_dataset(
    gene_wise_data: dict,
    species_metadata: dict,
    media_vectorizer: MediaVectorizer,
) -> Dataset:
    """Map the internal table onto schema.py objects.

    genome_id  = species key                       organism from species_metadata (else id)
    Gene       = aa/dna/promoter sequences         contig/strand/coords default (they live in
                                                   cds_meta.json, not the expression table)
    Medium     = unique (name, vector, params)     composition reconstructed from the vocabulary
    ExpressionRecord = one (genome, gene, medium)  expression = log-TPM (None dropped)
    """
    genomes: List[Genome] = []
    media_by_key: Dict[tuple, Medium] = {}
    expression: List[ExpressionRecord] = []

    for gid, genes in gene_wise_data.items():
        meta = species_metadata.get(gid, {})
        organism = meta.get("organism") or gid   # organism resolved by split.py from metadata CSV
        gene_objs: List[Gene] = []

        for gene_id, gd in genes.items():
            gi = gd["gene_info"]
            gene_objs.append(Gene(
                gene_id=gene_id,
                aa_sequence=gi.get("gene_sequence_aa") or "",
                dna_sequence=gi.get("gene_sequence_dna") or "",
                promoter_sequence=gi.get("promoter_sequence"),
                # contig/strand/start/end come from cds_meta.json (feature stage), not here
            ))

            for c in gd["conditions"]:
                expr = c.get("expression")
                if expr is None:
                    continue
                name = c.get("media_name", "")
                vec = c.get("media_vector")
                params = _numeric_params(c.get("condition_metadata", {}))

                vec_list = np.asarray(vec, dtype=np.float32).tolist() if vec is not None else None
                mkey = (name, tuple(vec_list) if vec_list is not None else None,
                        tuple(sorted(params.items())))
                if mkey not in media_by_key:
                    medium_id = name if len(media_by_key) == 0 or name not in \
                        {m.medium_id for m in media_by_key.values()} else f"{name}#{len(media_by_key)}"
                    media_by_key[mkey] = Medium(
                        medium_id=medium_id,
                        composition=(media_vectorizer.vec_to_composition(vec)
                                     if vec is not None else {}),
                        vector=vec_list,
                        params=params,
                    )
                medium_id = media_by_key[mkey].medium_id
                expression.append(ExpressionRecord(gid, gene_id, medium_id, float(expr)))

        genomes.append(Genome(genome_id=gid, organism=organism, genes=gene_objs))

    return Dataset(genomes=genomes, media=list(media_by_key.values()), expression=expression)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── shared vocabularies (optional feature lists + supplements) ──
    bgc_local_regulator_list = load_lines(os.path.join(SHARED_LIST_DIR, "whole_bgc_local_regulator_list.txt"))
    bgc_tfbs_list            = load_lines(os.path.join(SHARED_LIST_DIR, "whole_bgc_tfbs_list.txt"))
    whole_ec_list            = load_lines(os.path.join(SHARED_LIST_DIR, "whole_ec_list.txt"))
    whole_duf_list           = load_lines(os.path.join(SHARED_LIST_DIR, "whole_regulatory_duf_list.txt"))
    whole_pfam_list          = load_lines(os.path.join(SHARED_LIST_DIR, "whole_regulatory_pfam_list.txt"))
    whole_supplement_list    = load_lines(os.path.join(SHARED_LIST_DIR, "whole_supplement_list.txt"))
    whole_bgc_type_list      = load_lines(os.path.join(SHARED_LIST_DIR, "whole_bgc_type_list.txt"))
    pfam2go = load_pfam2go(PFAM2GO_PATH, TARGET_GOS)

    # ── media vectorizer (both sources -> one fixed vocabulary) ──
    media_vectorizer = MediaVectorizer(
        csv_path=DTU_MEDIA_CSV, json_path=MANUAL_MEDIA_JSON,
        supplement_list=whole_supplement_list)
    print(f"Media vocabulary dim: {media_vectorizer.component_dim}")

    shared_feature_kwargs = dict(
        pfam2go=pfam2go, whole_duf_list=whole_duf_list, whole_pfam_list=whole_pfam_list,
        whole_ec_list=whole_ec_list, bgc_local_regulator_list=bgc_local_regulator_list,
        bgc_tfbs_list=bgc_tfbs_list, whole_bgc_type_list=whole_bgc_type_list)

    # ── DTU_Pep ──
    dtu_promoter_seqs = load_promoter_seqs(DTU_PROMOTER_FASTA)
    dtu_operon_df = pd.read_csv(DTU_OPERON_TSV, sep="\t")
    dtu_operon_df["sequence"] = dtu_operon_df["operon_id"].map(dtu_promoter_seqs)
    dtu_species = [x.replace("_antismash", "") for x in os.listdir(DTU_GENOME_DIR)
                   if x.endswith("_antismash")]

    gwd_dtu, meta_dtu = build_gene_wise_data_dtu(
        species_list=dtu_species, genome_dir=DTU_GENOME_DIR, tpm_dir=DTU_TPM_DIR,
        pfam_results_dir=DTU_PFAM_DIR, deepec_results_dir=DTU_DEEPEC_DIR,
        operon_df=dtu_operon_df, media_vectorizer=media_vectorizer, **shared_feature_kwargs)

    # ── Manual_curated ──
    man_promoter_seqs = load_promoter_seqs(MANUAL_PROMOTER_FASTA)
    man_operon_df = pd.read_csv(MANUAL_OPERON_TSV, sep="\t")
    man_operon_df["sequence"] = man_operon_df["operon_id"].map(man_promoter_seqs)
    man_species = [x.replace(".gbk", "") for x in os.listdir(MANUAL_GENOME_DIR)
                   if x.endswith(".gbk")]

    gwd_man, meta_man = build_gene_wise_data_manual(
        species_list=man_species, genome_dir=MANUAL_GENOME_DIR, tpm_dir=MANUAL_TPM_DIR,
        pfam_results_dir=MANUAL_PFAM_DIR, deepec_tsv_path=MANUAL_DEEPEC_TSV,
        operon_df=man_operon_df, metadata_csv_path=MANUAL_METADATA_CSV,
        media_vectorizer=media_vectorizer, media_json_path=MANUAL_MEDIA_JSON,
        **shared_feature_kwargs)

    # ── merge (Manual wins on name collision) ──
    merged_gwd = {**gwd_dtu, **gwd_man}
    merged_meta = {**meta_dtu, **meta_man}
    for sp in set(gwd_dtu) & set(gwd_man):
        merged_gwd[sp] = gwd_man[sp]
        merged_meta[sp] = meta_man[sp]
    print(f"Merged: {len(merged_gwd)} species "
          f"(DTU={len(gwd_dtu)}, Manual={len(gwd_man)})")

    # ── save internal table (consumed downstream by data.load_reference) ──
    os.makedirs(os.path.dirname(INTERNAL_PKL), exist_ok=True)
    with open(INTERNAL_PKL, "wb") as f:
        pickle.dump({
            "gene_wise_data": merged_gwd,
            "species_metadata": merged_meta,
            "component_list": media_vectorizer.component_list,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved internal table -> {INTERNAL_PKL}")

    # ── convert to the standard schema + save ──
    dataset = to_standard_dataset(merged_gwd, merged_meta, media_vectorizer)
    save_dataset(dataset, DATASET_DIR)
    print(f"Saved standard-schema Dataset -> {DATASET_DIR}/  "
          f"(genomes={len(dataset.genomes)}, media={len(dataset.media)}, "
          f"expression={len(dataset.expression)})")

    # NOTE: the leakage-safe genus train/val/test split is a separate stage — see
    # repository root/split.py (it derives genus from the metadata CSVs in config.py).


if __name__ == "__main__":
    main()
