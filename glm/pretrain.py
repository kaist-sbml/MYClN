#!/usr/bin/env python
"""Self-supervised pretraining loop for the ActinoBERT backbone.

Operon-masked gene reconstruction over the full genome corpus (batch_size=1,
one genome per step). Loss = MSE + CONTRASTIVE_WEIGHT * InfoNCE +
BGC_LOSS_WEIGHT * BGC-focal, with the BGC term gated by USE_BGC_LOSS.

Deployed recipe: cis + topology model, deep DNA cis-encoder (CIS_DIM=96,
CIS_DIL=4), optional warm-start from a topology-only backbone, 1 epoch over the
full corpus (~19 h on an A100). Checkpoints are written every LOG_EVERY genomes
(resumable). All model classes / losses / data layer come from model.py.
"""
import os, sys, time, glob, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as release_config
from glm.model import *   # constants, losses, HierGenomeDataset/collate_fn, build_model, build_topo_only

# ═══════════════════════════ config (deployed recipe) ═══════════════════════
SAVE = release_config.GLM_OUTPUT_DIR + os.sep                       # checkpoints / stats
ROOT = release_config.GENOME_CORPUS     # genome corpus
MAX_GENES_TRAIN = 13000        # per-genome gene cap
SUBSET          = 0            # 0 = full corpus; >0 = random subset (fast smoke test)
NUM_WORKERS     = 0            # keep 0: CUDA-after-fork with model on GPU deadlocks
TOPO_ONLY       = os.environ.get("GLM_TOPO_ONLY", "0") == "1"        # False = cis + topology (deployed); True = topology-only stage-1
USE_BGC_LOSS    = False        # ★ BGC 3-class focal auxiliary term ON/OFF.  DEPLOYED = False.
                               #   The downstream model is BGC-free, and dropping this loss is
                               #   the consistent choice: pooled r is a statistical TIE (Δ=-0.0003,
                               #   95% CI [-0.008,+0.008]) and BGC-off is nominally better on
                               #   condition-response (resp 0.239 vs 0.216). Simpler + no BGC labels.
                               #   Set True to reproduce the older BGC-on checkpoint.
CKPT_NAME       = os.environ.get("GLM_CKPT_NAME", "v7_cisfull_bgcoff")  # SAVE/{CKPT_NAME}{,_best,_ep<N>}.pth
WARM_FROM       = release_config.GLM_WARM_START   # optional backbone warm-start (set FROM_SCRATCH=True to skip)
FROM_SCRATCH    = os.environ.get("GLM_FROM_SCRATCH", "0") == "1"        # True = random init (no warm-start)
CIS_DIM, CIS_DIL = 96, 4       # deep nucleotide cis-encoder (the key lever)
NEW_LR_MULT     = 15           # LR multiplier for freshly-initialised modules
BASE_LR         = LEARNING_RATE  # from model.py constants (5e-5)
EPOCHS          = int(os.environ.get("GLM_EPOCHS", "1"))            # 1 epoch over the full corpus (~19 h)
MAX_OP_TRAIN    = 6000         # skip genomes above this many operons (OOM guard)
LOG_EVERY       = 100          # progress + checkpoint interval (genomes)
# ════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("gLM pretraining requires CUDA")
    os.makedirs(SAVE, exist_ok=True)
    dev = torch.device("cuda"); set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    def log(*a): print(*a, flush=True)
    t0 = time.time()
    
    ds = HierGenomeDataset(ROOT, max_genes=MAX_GENES_TRAIN, max_operons=MAX_OPERONS, promoter_len=PROMOTER_LEN)
    log(f"dataset: {len(ds)} genomes")
    mean, std = np.load(release_config.EMB_STATS)["mean"], np.load(release_config.EMB_STATS)["std"]
    ds.set_emb_stats(mean, std)
    # global dim
    g0 = ds[0]; global_dim = g0["global_features"].shape[0]; log(f"global_dim={global_dim}")
    if 0 < SUBSET < len(ds):
        gsub = torch.Generator().manual_seed(123); perm = torch.randperm(len(ds), generator=gsub)[:SUBSET].tolist()
        ds = Subset(ds, perm); log(f"[subset] using {len(ds)} genomes")
    NW = NUM_WORKERS
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=NW, collate_fn=collate_fn,
                        pin_memory=False, persistent_workers=(NW > 0), prefetch_factor=(4 if NW > 0 else None))
    # resume from own checkpoint if present, else warm-start from the configured backbone
    BASE = CKPT_NAME
    WARMSRC = WARM_FROM
    RESUME = SAVE + BASE + ".pth"
    if os.path.exists(RESUME):   WARM = RESUME                # resume own checkpoint if present
    elif FROM_SCRATCH:           WARM = None                  # random init
    elif WARMSRC:                WARM = WARMSRC
    else:                        WARM = None
    if TOPO_ONLY:
        model = build_topo_only(global_dim, warm_start=WARM).to(dev)
    else:
        model = build_model(global_dim, warm_start=WARM,
            promoter_conv_dim=CIS_DIM, promoter_dils=CIS_DIL,
            promoter_use_ckpt=False).to(dev)   # ckpt OFF: 2x faster, no memory cost (single-chunk genomes)
    log(f"[pretrain] mode={'TOPO_ONLY' if TOPO_ONLY else 'cis+topo'} base={BASE} "
        f"warm={os.path.basename(WARM) if WARM else 'SCRATCH(random init)'} ({'RESUME' if WARM == RESUME else 'fresh'})")
    # ★ new topology/cis modules init near 0; a warm-started backbone sits at a good optimum -> weak
    #   pressure to use them. Give the fresh modules a higher LR so they actually learn.
    #   TOPO_ONLY -> the topology modules are fresh; cis mode -> the cis-encoder is fresh (topology
    #   modules are already warm-started from stage-1 -> base LR).
    _newkeys = ('topology_emb', 'rel_pos_bias', 'enddist_proj', 'register_tokens', 'contig_proj') if TOPO_ONLY else ('promoter_encoder',)
    _new  = [p for n, p in model.named_parameters() if any(k in n for k in _newkeys)]
    _base = [p for n, p in model.named_parameters() if not any(k in n for k in _newkeys)]
    opt = torch.optim.AdamW([{"params": _base, "lr": BASE_LR},
                             {"params": _new,  "lr": BASE_LR * NEW_LR_MULT}], weight_decay=WEIGHT_DECAY)
    log(f"[opt] base_lr={BASE_LR} new_module_lr={BASE_LR * NEW_LR_MULT} (new tensors={len(_new)})")
    scaler = GradScaler()
    log(f"params={sum(p.numel() for p in model.parameters()):,}")
    
    def move(b):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    ACC = ACCUMULATION_STEPS
    best = 1e9; step = 0; n_oom = 0
    for ep in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad(set_to_none=True); run = 0.; nb = 0; te = time.time()
        for gi, batch in enumerate(loader):
            if batch is None: continue
            b = move(batch)
            valid = ~b["gene_padding_mask"]
            if valid.sum() < 8: continue
            if int(b["operon_valid_mask"].sum()) > MAX_OP_TRAIN: continue   # OOM guard: skip huge genomes
            mpos = make_mask_positions(b["gene_padding_mask"], b["operon_id_of_gene"], MASK_RATIO, NEIGHBOR_MASK_PROB, dev)
            m = mpos & valid
            if m.sum() < 1: continue
            try:
                with autocast():
                    out = model(gene_emb=b["gene_emb"], gene_padding_mask=b["gene_padding_mask"], operon_id_of_gene=b["operon_id_of_gene"],
                                strand_idx=b["strand_idx"], intergenic_spacing=b["intergenic_spacing"], promoter_onehot=b["promoter_onehot"],
                                promoter_numeric_features=b["promoter_numeric_features"], operon_valid_mask=b["operon_valid_mask"],
                                operon_sizes=b["operon_sizes"], global_features=b["global_features"], masked_gene_positions=mpos,
                                operon_contig_start=b["operon_contig_start"], operon_contig_len=b["operon_contig_len"], operon_topology=b["operon_topology"])
                    target = b["gene_emb"]; recon = out["recon_gene"]
                    mse = F.mse_loss(recon[m], target[m])
                    con = contrastive_mlm_loss(recon, target, mpos, valid, CONTRASTIVE_TEMP)
                    ov = b["operon_valid_mask"][0]
                    bgc_loss = torch.tensor(0., device=dev)
                    if USE_BGC_LOSS and ov.any():
                        tgt = bgc_score_to_class(b["operon_bgc_score"][0][ov])
                        bgc_loss = bgc_focal_loss(out["bgc_pred"][0][ov], tgt, BGC_CLASS_WEIGHTS)
                    loss = (mse + CONTRASTIVE_WEIGHT * con + (BGC_LOSS_WEIGHT if USE_BGC_LOSS else 0.0) * bgc_loss) / ACC
                scaler.scale(loss).backward(); run += loss.item() * ACC; nb += 1
                if (gi + 1) % ACC == 0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                if isinstance(ex, torch.cuda.OutOfMemoryError) or "out of memory" in str(ex).lower():
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); n_oom += 1; continue
                raise
            if (gi + 1) % LOG_EVERY == 0:
                avg = run / max(nb, 1); log(f"  ep{ep} {gi + 1}/{len(ds)} loss={avg:.4f} oom_skip={n_oom} ({time.time() - te:.0f}s)")
                torch.save(model.state_dict(), SAVE + BASE + ".pth")
                if avg < best: best = avg; torch.save(model.state_dict(), SAVE + BASE + "_best.pth")
                run = 0.; nb = 0
        torch.save(model.state_dict(), SAVE + f"{BASE}_ep{ep}.pth")
        log(f"[epoch {ep}] done ({time.time() - te:.0f}s)  saved {BASE}_ep{ep}.pth")
    log(f"pretraining done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
