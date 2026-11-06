#!/usr/bin/env python
"""Per-genome gLM feature inputs — operon layout, promoter features, global stats.

For every genome directory this computes the topology/cis features the genomic
language model (gLM) consumes and writes them next to the ESM embeddings as
`operon_meta.npz`. It matches each called gene to its operon (from an operon-
prediction table), one-hot-encodes and scores each operon's promoter, and derives
per-gene / per-genome numeric features.

Per genome directory (under ROOT_GENOME_DIR):
  INPUT   cds_emb.npy      (G, 480)   external ESM-2 per-CDS embeddings
  INPUT   cds_meta.json    list[{contig, start, end, strand, locus_tag/gene_id}]
  OUTPUT  operon_meta.npz  arrays:
            operon_id_of_gene          (G,)              gene -> operon index
            gene_strand                (G,)              +1 / -1
            intergenic_upstream        (G,)              bp to previous gene
            gene_role_type             (G,)              reserved (zeros)
            promoter_onehot            (O, PROMOTER_LEN, 5)
            promoter_numeric_features  (O, 12)           sigma-factor-like motif scores
            operon_valid_mask          (O,)
            operon_sizes               (O,)
            operon_first_gene_idx      (O,)
            global_features            (12,)             per-genome summary

Global inputs (shared across genomes):
  OPERON_TSV      operon predictions   (genome, contig, operon_id, strand, start, end, genes)
  PROMOTER_FASTA  operon promoter sequences  (header: genome|contig|operon_id|...)

External dependency: cds_emb.npy / cds_meta.json come from an external gene caller
+ ESM-2 pass; the operon table / promoter FASTA come from an external operon
predictor. This step is NOT runnable without those raw inputs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as release_config
import os
import json
import glob
import pickle
import traceback
import numpy as np
import pandas as pd
from Bio import SeqIO
from bisect import bisect_right
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==========================================
# 0. Paths / config (real artifact locations; configured via config.py)
# ==========================================
ROOT_GENOME_DIR = os.path.join(release_config.RAW_DATA_DIR, 'Whole_genome_data/cds_embeddings')
OPERON_TSV      = os.path.join(release_config.RAW_DATA_DIR, 'Whole_genome_data/operons_t0p5.tsv')
PROMOTER_FASTA  = os.path.join(release_config.RAW_DATA_DIR, 'Whole_genome_data/promoters_up250_down50_t0p5.fasta')

PROMOTER_LEN   = 300
NUM_WORKERS    = max(1, cpu_count() - 1)
SAVE_EVERY     = 50
RESUME         = True
SKIP_IF_EXISTS = False

CHECKPOINT_PATH = os.path.join(ROOT_GENOME_DIR, "_operon_meta_build_checkpoint.pkl")
ERROR_LOG_PATH  = os.path.join(ROOT_GENOME_DIR, "_operon_meta_build_errors.log")


# ==========================================
# 1. Small utilities
# ==========================================
def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def save_checkpoint(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def append_error_log(msg):
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(msg + "\n")

# ──────────────────────────────────────────
# DNA / sequence utilities
# ──────────────────────────────────────────
_DNA_MAP = np.zeros((256, 5), dtype=np.float32)
for _c, _v in zip(b"ACGT", range(4)):
    _DNA_MAP[_c][_v] = 1.0
_DNA_MAP[ord("N")] = [0, 0, 0, 0, 1]

def dna_onehot(seq: str, L: int = 300) -> np.ndarray:
    seq = seq.upper().replace("U", "T")
    arr = np.frombuffer(seq[:L].encode("ascii"), dtype=np.uint8)
    out = np.zeros((L, 5), dtype=np.float32)
    out[:len(arr)] = _DNA_MAP[arr]
    return out

def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    a = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    return float(np.isin(a, [ord("G"), ord("C")]).mean())

def at_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    a = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    return float(np.isin(a, [ord("A"), ord("T")]).mean())

def _slide_match(seq_u8: np.ndarray, motif_u8: np.ndarray,
                 start: int, stop: int) -> tuple:
    mlen = len(motif_u8)
    if stop < start + mlen:
        return 0.0, -1
    windows = np.lib.stride_tricks.sliding_window_view(seq_u8[start:stop + mlen], mlen)
    scores  = (windows == motif_u8).mean(axis=1)
    best    = int(np.argmax(scores))
    return float(scores[best]), start + best

def scan_sigma_like_features(seq: str) -> np.ndarray:
    """Score -10 (TATAAT) / -35 (TTGACA) sigma-factor-like boxes + spacer + GC/AT."""
    seq    = seq.upper().replace("U", "T")
    n      = len(seq)
    seq_u8 = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    m10 = np.frombuffer(b"TATAAT", dtype=np.uint8)
    m35 = np.frombuffer(b"TTGACA", dtype=np.uint8)

    reg_s = max(0, n - 120)
    reg_e = n - 6

    bm10, pm10 = _slide_match(seq_u8, m10, reg_s, reg_e)
    bm35, pm35 = _slide_match(seq_u8, m35, reg_s, reg_e)

    best_pair   = -1.0
    best_spacer = -1

    if reg_e >= reg_s + 6:
        wins35  = np.lib.stride_tricks.sliding_window_view(seq_u8[reg_s:reg_e + 6], 6)
        s35_arr = (wins35 == m35).mean(axis=1)

        for sp in range(15, 20):
            pos10_start = reg_s + 6 + sp
            pos10_end   = reg_e + 6 + sp
            if pos10_end > n:
                break
            wins10  = np.lib.stride_tricks.sliding_window_view(seq_u8[pos10_start:pos10_end], 6)
            s10_arr = (wins10 == m10).mean(axis=1)
            min_len = min(len(s35_arr), len(s10_arr))
            pairs   = 0.5 * s35_arr[:min_len] + 0.5 * s10_arr[:min_len]
            idx     = int(np.argmax(pairs))
            if pairs[idx] > best_pair:
                best_pair   = float(pairs[idx])
                best_spacer = sp

    last50   = seq[-50:]  if n >= 50  else seq
    first100 = seq[:100]  if n >= 100 else seq

    return np.array([
        gc_fraction(seq),
        at_fraction(seq),
        max(bm10, 0.0),
        max(bm35, 0.0),
        max(best_pair, 0.0),
        float(max(best_spacer, 0)),
        float(max(pm10, 0)),
        float(max(pm35, 0)),
        gc_fraction(last50),
        gc_fraction(first100),
        float(seq.count("TTT")),
        float(seq.count("AAA")),
    ], dtype=np.float32)


# ==========================================
# 2. Global data loaders
# ==========================================
def load_operon_table(tsv_path: str):
    flush_print("Loading operon table...")
    df = pd.read_csv(tsv_path, sep="\t")

    for c in ("genome", "contig", "operon_id", "strand", "start", "end", "genes"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["strand"] = df["strand"].astype(int)
    df["start"]  = df["start"].astype(int)
    df["end"]    = df["end"].astype(int)

    operon_by_genome: dict = defaultdict(lambda: defaultdict(list))
    for _, r in df.iterrows():
        operon_by_genome[r["genome"]][(r["contig"], int(r["strand"]))].append({
            "contig":    r["contig"],
            "operon_id": r["operon_id"],
            "strand":    int(r["strand"]),
            "start":     int(r["start"]),
            "end":       int(r["end"]),
            "genes":     str(r["genes"]),
        })

    starts_cache: dict = {}
    for genome, groups in operon_by_genome.items():
        for key, ops in groups.items():
            ops.sort(key=lambda x: (x["start"], x["end"]))
            starts_cache[(genome, key)] = [op["start"] for op in ops]

    # defaultdict -> plain dict (better copy-on-write efficiency after fork)
    operon_by_genome = {g: dict(v) for g, v in operon_by_genome.items()}

    flush_print(f"Loaded {len(operon_by_genome):,} genomes from operon table.")
    return operon_by_genome, starts_cache


def load_promoter_fasta(fasta_path: str) -> dict:
    flush_print("Loading promoter FASTA...")
    nested: dict = defaultdict(dict)
    count = 0
    with open(fasta_path) as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            parts = rec.description.strip().split("|")
            if len(parts) < 3:
                continue
            genome, contig, op_id = parts[0], parts[1], parts[2]
            nested[genome][(contig, op_id)] = str(rec.seq)
            count += 1
    flush_print(f"Loaded {count:,} promoter entries.")
    return dict(nested)


# ==========================================
# 3. Worker — initializer style
#    Called once at Pool creation; workers read shared data via globals.
# ==========================================
_OPERON_BY_GENOME = None
_STARTS_CACHE     = None
_PROMOTER_NESTED  = None

def _init_worker(operon_by_genome, starts_cache, promoter_nested):
    global _OPERON_BY_GENOME, _STARTS_CACHE, _PROMOTER_NESTED
    _OPERON_BY_GENOME = operon_by_genome
    _STARTS_CACHE     = starts_cache
    _PROMOTER_NESTED  = promoter_nested


def process_genome(gdir: str):
    """Returns ("success"|"skipped"|"failed", gdir, error_msg_or_None)."""
    accession = os.path.basename(gdir)
    save_path = os.path.join(gdir, "operon_meta.npz")

    if SKIP_IF_EXISTS and os.path.exists(save_path):
        return ("skipped", gdir, None)

    try:
        cds_path  = os.path.join(gdir, "cds_emb.npy")
        meta_path = os.path.join(gdir, "cds_meta.json")

        if not os.path.exists(cds_path) or not os.path.exists(meta_path):
            return ("skipped", gdir, None)

        cds_emb = np.load(cds_path)
        if cds_emb.ndim != 2 or cds_emb.shape[1] != 480:
            raise ValueError(f"Invalid cds_emb shape: {cds_emb.shape}")

        with open(meta_path) as f:
            genes = json.load(f)
        if not isinstance(genes, list):
            raise ValueError("cds_meta.json is not a list")

        gene_contig = np.array([g["contig"]      for g in genes], dtype=object)
        gene_start  = np.array([int(g["start"])  for g in genes], dtype=np.int64)
        gene_end    = np.array([int(g["end"])    for g in genes], dtype=np.int64)
        gene_strand = np.array([int(g["strand"]) for g in genes], dtype=np.int64)

        G = len(gene_start)
        if G != len(cds_emb):
            diff = abs(G - len(cds_emb))
            if diff > 5:
                raise ValueError(f"meta/emb count mismatch: {G} vs {len(cds_emb)}")
            mn          = min(G, len(cds_emb))
            cds_emb     = cds_emb[:mn]
            gene_contig = gene_contig[:mn]
            gene_start  = gene_start[:mn]
            gene_end    = gene_end[:mn]
            gene_strand = gene_strand[:mn]
            G = mn

        # ── gene -> operon matching (bisect) ──────────────
        acc_ops      = _OPERON_BY_GENOME.get(accession, {})
        raw_operon   = []
        singleton_ct = 0

        for i in range(G):
            key  = (gene_contig[i], int(gene_strand[i]))
            ops  = acc_ops.get(key, [])
            found = None

            if ops:
                sc  = _STARTS_CACHE.get((accession, key), [])
                idx = bisect_right(sc, int(gene_start[i])) - 1
                if idx >= 0:
                    op = ops[idx]
                    if gene_start[i] >= op["start"] and gene_end[i] <= op["end"]:
                        found = op["operon_id"]

            if found is None:
                found = f"singleton_{singleton_ct}"
                singleton_ct += 1
            raw_operon.append(found)

        # operon label -> int id
        seen   = {}
        unique = []
        for x in raw_operon:
            if x not in seen:
                seen[x] = len(unique)
                unique.append(x)

        operon_id_of_gene = np.array([seen[x] for x in raw_operon], dtype=np.int64)
        num_operons       = len(unique)

        # ── intergenic spacing ───────────────────────────
        intergenic    = np.zeros(G, dtype=np.float32)
        intergenic[0] = 1000.0
        for i in range(1, G):
            if gene_contig[i] != gene_contig[i - 1]:
                intergenic[i] = 1000.0
            else:
                intergenic[i] = float(max(0, gene_start[i] - gene_end[i - 1]))

        gene_role_type = np.zeros(G, dtype=np.int64)

        # ── operon size / first gene index ───────────────
        operon_sizes          = np.zeros(num_operons, dtype=np.int64)
        operon_first_gene_idx = np.zeros(num_operons, dtype=np.int64)

        np.add.at(operon_sizes, operon_id_of_gene, 1)

        first_seen_set: set = set()
        for gi, oi in enumerate(operon_id_of_gene):
            if oi not in first_seen_set:
                operon_first_gene_idx[oi] = gi
                first_seen_set.add(oi)

        # ── promoter features ────────────────────────────
        promoter_onehot   = np.zeros((num_operons, PROMOTER_LEN, 5), dtype=np.float32)
        promoter_numeric  = np.zeros((num_operons, 12),              dtype=np.float32)
        operon_valid_mask = np.zeros(num_operons, dtype=bool)

        acc_promoters = _PROMOTER_NESTED.get(accession, {})

        for op_label, op_idx in seen.items():
            first_idx = operon_first_gene_idx[op_idx]
            contig    = gene_contig[first_idx]

            if op_label.startswith("singleton_"):
                seq = None
            else:
                short_id = op_label.split("|")[-1]
                seq = acc_promoters.get((contig, short_id), None)

            if seq is None:
                seq = "N" * PROMOTER_LEN

            promoter_onehot[op_idx]   = dna_onehot(seq, PROMOTER_LEN)
            promoter_numeric[op_idx]  = scan_sigma_like_features(seq)
            operon_valid_mask[op_idx] = True

        # ── global features ──────────────────────────────
        plus_frac  = float(np.mean(gene_strand > 0)) if G else 0.0
        minus_frac = float(np.mean(gene_strand < 0)) if G else 0.0

        global_features = np.array([
            float(np.log1p(G)),
            float(np.log1p(num_operons)),
            float(np.mean(intergenic))            if G           else 0.0,
            float(np.std(intergenic))             if G           else 0.0,
            plus_frac,
            minus_frac,
            float(np.mean(operon_sizes))          if num_operons else 0.0,
            float(np.std(operon_sizes))           if num_operons else 0.0,
            float(np.mean(promoter_numeric[:,0])) if num_operons else 0.0,
            float(np.mean(promoter_numeric[:,2])) if num_operons else 0.0,
            float(np.mean(promoter_numeric[:,3])) if num_operons else 0.0,
            float(np.mean(promoter_numeric[:,4])) if num_operons else 0.0,
        ], dtype=np.float32)

        # ── save (compressed) ────────────────────────────
        np.savez_compressed(
            save_path,
            operon_id_of_gene         = operon_id_of_gene,
            gene_strand               = gene_strand,
            intergenic_upstream       = intergenic,
            gene_role_type            = gene_role_type,
            promoter_onehot           = promoter_onehot,
            promoter_numeric_features = promoter_numeric,
            operon_valid_mask         = operon_valid_mask,
            global_features           = global_features,
            operon_sizes              = operon_sizes,
            operon_first_gene_idx     = operon_first_gene_idx,
        )

        return ("success", gdir, None)

    except Exception as e:
        msg = f"[ERROR] {gdir}: {repr(e)}\n{traceback.format_exc()}"
        return ("failed", gdir, msg)


# ==========================================
# 4. Main
# ==========================================
def main():
    operon_by_genome, starts_cache = load_operon_table(OPERON_TSV)
    promoter_nested                = load_promoter_fasta(PROMOTER_FASTA)

    genome_dirs = sorted([
        d for d in glob.glob(os.path.join(ROOT_GENOME_DIR, "*"))
        if os.path.isdir(d)
    ])
    flush_print(f"Total genomes: {len(genome_dirs):,}")

    # ── resume ───────────────────────────────────────────
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        flush_print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt           = load_checkpoint(CHECKPOINT_PATH)
        processed_dirs = set(ckpt.get("processed_dirs", []))
        success        = int(ckpt.get("success", 0))
        failed         = int(ckpt.get("failed",  0))
        skipped        = int(ckpt.get("skipped", 0))
        flush_print(
            f"Resumed: processed={len(processed_dirs):,} "
            f"success={success} failed={failed} skipped={skipped}"
        )
    else:
        processed_dirs = set()
        success = failed = skipped = 0

    remaining = [d for d in genome_dirs if d not in processed_dirs]
    flush_print(f"Remaining: {len(remaining):,}  |  Workers: {NUM_WORKERS}")

    if not remaining:
        flush_print("Nothing to do.")
        return

    # ── Pool: shared data passed once via initializer ────
    # On Linux (fork) the shared tables are copy-on-write, so no IPC cost.
    with Pool(
        processes   = NUM_WORKERS,
        initializer = _init_worker,
        initargs    = (operon_by_genome, starts_cache, promoter_nested),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_genome, remaining, chunksize=8),
            total = len(remaining),
            desc  = "Building operon_meta",
        )
        for status, gdir, err_msg in pbar:
            if status == "success":
                success += 1
            elif status == "failed":
                failed += 1
                if err_msg:
                    flush_print(err_msg)
                    append_error_log(err_msg)
            else:
                skipped += 1

            processed_dirs.add(gdir)
            pbar.set_postfix(success=success, failed=failed, skipped=skipped)

            if len(processed_dirs) % SAVE_EVERY == 0:
                save_checkpoint(
                    {
                        "processed_dirs": sorted(processed_dirs),
                        "success": success,
                        "failed":  failed,
                        "skipped": skipped,
                    },
                    CHECKPOINT_PATH,
                )

    # ── final checkpoint ─────────────────────────────────
    save_checkpoint(
        {
            "processed_dirs": sorted(processed_dirs),
            "success": success,
            "failed":  failed,
            "skipped": skipped,
        },
        CHECKPOINT_PATH,
    )

    flush_print("\n=== Done ===")
    flush_print(f"Success : {success:,}")
    flush_print(f"Failed  : {failed:,}")
    flush_print(f"Skipped : {skipped:,}")
    flush_print(f"Checkpoint : {CHECKPOINT_PATH}")
    flush_print(f"Error log  : {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()
