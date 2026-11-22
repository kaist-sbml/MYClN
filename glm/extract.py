#!/usr/bin/env python
"""Extract per-gene embeddings with ActinoBERT, the MYClN genomic backbone.

Loads a trained checkpoint, runs the model over a genome corpus, and writes the
pickles consumed by the downstream expression model:

  bert_gene_embs{SUFFIX}.pkl     {species: (G, 960)}  per-gene [gene_in_operon | gene_operon_ctx]
  bert_gene_raw{SUFFIX}.pkl      {species: (G, 480)}  raw unstandardized ESM gene embeddings
  bert_cls_vectors{SUFFIX}.pkl   {species: (480,)}    genome CLS vector
  bert_gene_id_order{SUFFIX}.pkl {species: [gene_id]} gene order (matches the arrays above)

Input preprocessing mirrors HierGenomeDataset.__getitem__ (gene/operon
truncation, keep = operon_id < O, strand remap, intergenic log1p, promoter
numeric[:4], per-dim gene_emb standardization) plus per-operon topology.
"""
import os, sys, json, pickle, time
import numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as release_config
from glm.model import (build_model, build_topo_only,
                   EMB_DIM, MAX_OPERONS, PROMOTER_LEN, PROMOTER_NUMERIC_DIM)

# ═══════════════════════════════════ config ════════════════════════════════
SAVE_DIR = release_config.EXTRACT_OUTPUT_DIR + os.sep
ROOT_DIRS = release_config.GENOME_ROOTS
EMB_STATS_PATH = release_config.EMB_STATS
SPECIES_LIST_PKL = release_config.SPECIES_ORDER
CKPT = release_config.GLM_CHECKPOINT
OUT_SUFFIX = os.environ.get('GLM_OUT_SUFFIX', '_v7bgcoff')
DEVICE = os.environ.get('ACTEXPR_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
TOPO_ONLY        = False                                 # must match the checkpoint's model type
CIS_DIM, CIS_DIL = 96, 4                                 # MUST match the trained cis config
MAX_GENES        = 13000                                 # per-genome gene cap
# ════════════════════════════════════════════════════════════════════════════

dev      = torch.device(DEVICE)
PROM_NUM = PROMOTER_NUMERIC_DIM   # 4
PROM_LEN = PROMOTER_LEN           # 300


def find_genome_dir(species):
    for root in ROOT_DIRS:
        gd = os.path.join(root, species)
        if all(os.path.exists(os.path.join(gd, f))
               for f in ["cds_emb.npy", "operon_meta.npz", "cds_meta.json"]):
            return gd
    return None


def build_inputs(genome_dir, emb_mean, emb_std, device=None):
    """Dataset.__getitem__ preprocessing + surviving-gene original indices."""
    target_device = torch.device(device) if device is not None else dev
    gene_emb = np.load(os.path.join(genome_dir, "cds_emb.npy")).astype(np.float32)
    meta = np.load(os.path.join(genome_dir, "operon_meta.npz"), allow_pickle=True)

    operon_id = meta["operon_id_of_gene"].astype(np.int64)
    strand    = meta["gene_strand"].astype(np.int64)
    inter     = meta["intergenic_upstream"].astype(np.float32)
    prom_oh   = meta["promoter_onehot"].astype(np.float32)
    prom_num  = meta["promoter_numeric_features"].astype(np.float32)
    op_valid  = meta["operon_valid_mask"].astype(bool)
    op_sizes  = meta["operon_sizes"].astype(np.int64)
    glob_feat = meta["global_features"].astype(np.float32)

    # promoter_numeric -> first PROM_NUM(4) columns (pad if short)
    if prom_num.shape[1] >= PROM_NUM:
        prom_num = prom_num[:, :PROM_NUM]
    else:
        prom_num = np.pad(prom_num, ((0, 0), (0, PROM_NUM - prom_num.shape[1])))

    orig_idx = np.arange(len(gene_emb))
    n = min(len(gene_emb), len(operon_id), len(strand), len(inter))
    gene_emb, operon_id, strand, inter, orig_idx = (
        gene_emb[:n], operon_id[:n], strand[:n], inter[:n], orig_idx[:n])

    strand = np.where(strand < 0, 1, 2).astype(np.int64)
    inter  = np.log1p(np.clip(inter, 0, 5000)).astype(np.float32)

    O = min(len(prom_oh), MAX_OPERONS)
    prom_oh  = prom_oh[:O, :PROM_LEN, :]
    prom_num = prom_num[:O]
    op_valid = op_valid[:O]
    op_sizes = op_sizes[:O]

    if O > 0:
        keep = operon_id < O
        gene_emb, operon_id, strand, inter, orig_idx = (
            gene_emb[keep], operon_id[keep], strand[keep], inter[keep], orig_idx[keep])
    else:
        gene_emb = np.zeros((0, EMB_DIM), np.float32)
        operon_id = np.zeros((0,), np.int64); strand = np.zeros((0,), np.int64)
        inter = np.zeros((0,), np.float32); orig_idx = np.zeros((0,), np.int64)

    G = min(len(gene_emb), MAX_GENES)
    gene_emb, operon_id, strand, inter, orig_idx = (
        gene_emb[:G], operon_id[:G], strand[:G], inter[:G], orig_idx[:G])

    # per-dim standardization (identical to training)
    if G > 0:
        gene_emb = (gene_emb - emb_mean) / emb_std

    def t(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device=target_device).unsqueeze(0)

    inp = {
        "gene_emb":                  t(gene_emb, torch.float32),
        "gene_padding_mask":         torch.zeros(1, G, dtype=torch.bool, device=target_device),
        "operon_id_of_gene":         t(operon_id, torch.long),
        "strand_idx":                t(strand, torch.long),
        "intergenic_spacing":        t(inter, torch.float32),
        "promoter_onehot":           t(prom_oh, torch.float32),
        "promoter_numeric_features": t(prom_num, torch.float32),
        "operon_valid_mask":         t(op_valid, torch.bool),
        "operon_sizes":              t(op_sizes, torch.long),
        "global_features":           t(glob_feat, torch.float32),
        "masked_gene_positions":     None,
    }
    return inp, orig_idx, G, O


def add_topology(inp, genome_dir, O, device=None):
    """Attach per-operon topology (contig block start/len + linear/circular)."""
    target_device = torch.device(device) if device is not None else dev
    tp_path = os.path.join(genome_dir, "topology_meta.npz")
    if os.path.exists(tp_path):
        tm = np.load(tp_path)
        cs = tm["operon_contig_start"].astype(np.int64)[:O]
        cl = tm["operon_contig_len"].astype(np.int64)[:O]
        tp = tm["operon_topology"].astype(np.int64)[:O]
    else:                              # fallback: whole genome as one circular contig
        cs = np.zeros(O, np.int64); cl = np.full(O, max(O, 1), np.int64); tp = np.ones(O, np.int64)
    if O > 0:
        cl = np.minimum(cl, O - cs).clip(min=1)
    inp["operon_contig_start"] = torch.as_tensor(cs, dtype=torch.long, device=target_device).unsqueeze(0)
    inp["operon_contig_len"]   = torch.as_tensor(cl, dtype=torch.long, device=target_device).unsqueeze(0)
    inp["operon_topology"]     = torch.as_tensor(tp, dtype=torch.long, device=target_device).unsqueeze(0)
    return inp



def extract_one(genome_dir, checkpoint, stats_path, device='cpu'):
    """Run the exact final gLM over one prepared genome and return query arrays."""
    with np.load(stats_path) as stats:
        mean, std = stats['mean'].astype(np.float32), stats['std'].astype(np.float32)
    if mean.shape != (480,) or std.shape != (480,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('gLM normalization requires finite 480-d mean and positive std')
    inp, indices, G, O = build_inputs(genome_dir, mean, std, device=device)
    inp = add_topology(inp, genome_dir, O, device=device)
    with open(os.path.join(genome_dir, 'cds_meta.json')) as handle:
        cds = json.load(handle)
    if G != len(cds) or not np.array_equal(indices, np.arange(len(cds))):
        raise ValueError('The gLM caps/filtering would discard query genes; use a supported genome (<=13000 genes, <=9000 operons)')
    model = build_model(inp['global_features'].shape[-1], promoter_conv_dim=96,
                        promoter_dils=4, promoter_use_ckpt=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(**inp)
    ids = np.asarray([g['locus_tag'] for g in cds], dtype=str)
    arrays = {'esm_emb': np.load(os.path.join(genome_dir, 'cds_emb.npy')).astype(np.float32),
              'esm_order': ids, 'glm_order': ids.copy(),
              'glm_emb': torch.cat([out['gene_in_operon'][0], out['gene_operon_ctx'][0]], -1).float().cpu().numpy(),
              'glm_cls': out['cls'][0].float().cpu().numpy()}
    if any(not np.isfinite(arrays[k]).all() for k in ('esm_emb','glm_emb','glm_cls')):
        raise ValueError('Non-finite gLM output')
    return arrays

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    emb = np.load(EMB_STATS_PATH)
    emb_mean, emb_std = emb["mean"].astype(np.float32), emb["std"].astype(np.float32)
    order = pickle.load(open(SPECIES_LIST_PKL, "rb"))
    species = sorted(order.keys())

    # global dim from the first available genome
    gdim = 12
    for sp in species:
        gd = find_genome_dir(sp)
        if gd:
            gdim = int(np.load(os.path.join(gd, "operon_meta.npz"),
                               allow_pickle=True)["global_features"].shape[-1]); break

    if TOPO_ONLY:
        model = build_topo_only(gdim)
    else:   # cis mode: CIS_DIM/CIS_DIL MUST match the trained cis config
        model = build_model(gdim,
            promoter_conv_dim=CIS_DIM, promoter_dils=CIS_DIL, promoter_use_ckpt=False)
    # Extraction must load the entire trained architecture; partial warm-starts
    # are only appropriate during pretraining.
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(dev).eval()
    print(f"loaded {'topo-only' if TOPO_ONLY else 'cis+topo'} model from "
          f"{os.path.basename(CKPT)} | MAX_GENES={MAX_GENES} -> suffix='{OUT_SUFFIX}'", flush=True)

    gene_embs, raw_embs, cls_vecs, gid_order = {}, {}, {}, {}
    failures = []
    t0 = time.time()
    for sp in tqdm(species):
        gd = find_genome_dir(sp)
        if gd is None:
            failures.append(f"{sp}: missing genome directory"); continue
        try:
            inp, orig_idx, G, O = build_inputs(gd, emb_mean, emb_std)
            if G == 0:
                failures.append(f"{sp}: no surviving genes"); continue
            inp = add_topology(inp, gd, O)
            with torch.no_grad():
                out = model(**inp)
        except Exception as e:
            failures.append(f"{sp}: {e}"); continue
        gio, goc = out["gene_in_operon"][0], out["gene_operon_ctx"][0]
        gene_embs[sp] = torch.cat([gio, goc], -1).float().cpu().numpy()
        cls_vecs[sp]  = out["cls"][0].float().cpu().numpy()
        cds = np.load(os.path.join(gd, "cds_emb.npy")).astype(np.float32)
        raw_embs[sp]  = cds[orig_idx]
        cdsm = json.load(open(os.path.join(gd, "cds_meta.json")))
        gid_all = [g.get("locus_tag", g.get("gene_id", str(k))) for k, g in enumerate(cdsm)]
        gid_order[sp] = [gid_all[k] if k < len(gid_all) else str(k) for k in orig_idx.tolist()]

    if failures:
        raise RuntimeError("Extraction incomplete; no output pickles written: " + "; ".join(failures))
    if not gene_embs:
        raise ValueError("No species selected for extraction")
    for nm, obj in [("bert_gene_embs" + OUT_SUFFIX, gene_embs), ("bert_gene_raw" + OUT_SUFFIX, raw_embs),
                    ("bert_cls_vectors" + OUT_SUFFIX, cls_vecs), ("bert_gene_id_order" + OUT_SUFFIX, gid_order)]:
        pickle.dump(obj, open(SAVE_DIR + nm + ".pkl", "wb"), protocol=4)
    print(f"extracted {len(gene_embs)} species in {time.time() - t0:.0f}s -> *{OUT_SUFFIX}.pkl", flush=True)


if __name__ == "__main__":
    main()
