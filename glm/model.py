"""ActinoBERT -- the topology-aware genomic embedding backbone for MYClN.

Self-contained model definitions: constants, the self-supervised losses, the
`HierGenomeDataset`/`collate_fn` data layer, every nn.Module, and the build
functions. Consumed by `pretrain.py` (training) and `extract.py` (embedding
extraction), which do `from model import ...`.

Public model classes:
  ActinoBERT              -- top model: hierarchy + topology + deep DNA cis-encoder
  HierarchicalGenomeBERT  -- base hierarchy (cheap promoter encoder)
  PromoterCisEncoder      -- deep dilated-conv DNA cis-grammar encoder
Build functions:
  build_model(global_dim, warm_start=None, ...)  -- ActinoBERT (cis + topology)
  build_topo_only(global_dim, warm_start=None)   -- HierarchicalGenomeBERT (topology only)
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as release_config
import os
import glob
import csv
import random
import numpy as np
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ==========================================
# 1. 환경 설정 및 하이퍼파라미터
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.join(release_config.RAW_DATA_DIR, 'Whole_genome_data/cds_embeddings')

EMB_DIM   = 480
MODEL_DIM = 480

MAX_GENES            = 8092
MAX_OPERONS          = 9000
MAX_GENES_PER_OPERON = 24
OPERON_WINDOW_STRIDE = 12
PROMOTER_LEN         = 300

# 유효 피처 4개: GC(full), AT(full), GC_upstream, GC_tss_proximal
PROMOTER_NUMERIC_DIM = 4

LOCAL_WINDOW      = 64
MAX_SIZE_GLOBAL   = 64

BATCH_SIZE         = 1
ACCUMULATION_STEPS = 8
EPOCHS             = 50
LEARNING_RATE      = 5e-5
MASK_RATIO         = 0.15                       # ★ '오페론' 기준 마스킹 비율

NUM_STRAND_TYPES = 3

WEIGHT_DECAY    = 0.01
GRAD_CLIP_NORM  = 0.5
PATIENCE        = 8
NUM_WORKERS     = 2
SEED            = 42

BGC_LOSS_WEIGHT   = 0.1
BGC_WARMUP_EPOCHS = 2
USE_AMP = torch.cuda.is_available()

# BGC 3-class 설정
BGC_CLASS_BOUNDARIES = (0.1, 0.8)
BGC_CLASS_WEIGHTS    = torch.tensor([1.0, 18.0, 100.0])
BGC_FOCAL_GAMMA      = 2.0

# ── ★ 신규 설정 ────────────────────────────────────────────────
USE_EMB_STANDARDIZE = True
EMB_STATS_PATH      = "v5_emb_stats.npz"      # 차원별 mean/std 캐시 (train만으로 계산)
EMB_STATS_MAX_GENOMES = 300                    # 통계 추정에 사용할 train 유전체 수

CONTRASTIVE_WEIGHT  = 0.5                       # InfoNCE 가중 (0 이면 비활성)
CONTRASTIVE_TEMP    = 0.1
NEIGHBOR_MASK_PROB  = 0.3                       # ★ 선택 오페론의 인접(다음) 오페론 동시 마스킹 확률

MASH_CLUSTERS_TSV   = os.path.join(release_config.RAW_DATA_DIR, 'Whole_genome_data/mash_clusters.tsv')
VAL_FRAC            = 0.1


# ==========================================
# 2. Seed
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False



# ==========================================
# 3. 유틸
# ==========================================
def masked_mean(x, valid_mask, dim=1, eps=1e-8):
    w = valid_mask.float().unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp(min=eps)


def make_mask_positions(gene_padding_mask, operon_id_of_gene, mask_ratio,
                        neighbor_prob, device, generator=None):
    """
    ★ Operon 단위 마스킹.
      mask_ratio 비율의 '오페론'을 통째로 골라, 거기에 속한 모든 유전자를
      동시에 마스킹한다.
        - operon-mate 의 실제 ESM 임베딩으로 복원하는 지름길을 원천 차단.
        - 마스킹된 오페론은 genome transformer(operon-level context)를 거쳐
          '주변 오페론'으로부터 복원되도록 강제 → genome 맥락 학습 신호 강화.
      neighbor_prob:
        선택된 오페론의 genomic 인접(다음) 오페론도 확률적으로 함께 마스킹
        → 2개 이상 연속 오페론 공백을 만들어 더 긴 범위의 맥락 사용을 유도.
        (0 이면 단일 오페론 마스킹만)

      ※ mask_ratio 는 이제 '유전자'가 아니라 '오페론' 기준 비율이다.

    generator 를 지정하면 결정론적 마스킹(검증용 고정).
    """
    B, G           = gene_padding_mask.shape
    valid_mask     = ~gene_padding_mask
    mask_positions = torch.zeros_like(valid_mask)

    for b in range(B):
        vb = valid_mask[b]
        if not vb.any():
            continue
        op_ids_b = operon_id_of_gene[b]
        max_op   = int(op_ids_b[vb].max().item()) + 1

        # 유효 유전자를 가진 오페론만 마스킹 후보
        op_present = torch.zeros(max_op, dtype=torch.bool, device=device)
        op_present[op_ids_b[vb]] = True

        # 오페론 단위 베르누이 선택
        op_sel = (torch.rand(max_op, device=device, generator=generator)
                  < mask_ratio) & op_present
        if not op_sel.any():                       # 최소 1개 오페론 보장
            present_ids = torch.where(op_present)[0]
            j = torch.randint(0, present_ids.numel(), (1,), device=device,
                              generator=generator)
            op_sel[present_ids[j]] = True

        # 인접(다음) 오페론으로 마스킹 span 확장
        if neighbor_prob > 0.0 and max_op > 1:
            ext     = (torch.rand(max_op, device=device, generator=generator)
                       < neighbor_prob)
            nxt     = torch.zeros_like(op_sel)
            nxt[1:] = op_sel[:-1] & ext[1:]
            op_sel  = op_sel | (nxt & op_present)

        # 오페론 선택 → 유전자 단위로 환원 (패딩 유전자는 vb 로 제외)
        mask_positions[b] = op_sel[op_ids_b.clamp(0, max_op - 1)] & vb

    return mask_positions


def bgc_score_to_class(scores):
    lo, hi = BGC_CLASS_BOUNDARIES
    cls = torch.zeros_like(scores, dtype=torch.long)
    cls[scores >= lo] = 1
    cls[scores >= hi] = 2
    return cls


def bgc_focal_loss(logits, targets, class_weights, gamma=BGC_FOCAL_GAMMA):
    w  = class_weights.to(logits.device)
    ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce
    return focal.mean()


def contrastive_mlm_loss(recon, target, mask_positions, valid_mask, temp):
    """
    InfoNCE: 마스킹된 유전자의 복원 벡터가 (같은 유전체 내) 다른 모든
    유효 유전자 타깃보다 '자기 타깃'과 더 가깝도록 강제.
    → MSE 단독 사용 시 '평균 임베딩 예측' 으로 붕괴하는 문제 방지.

    recon          : (B, G, D)
    target         : (B, G, D)  — 표준화된 원본 gene_emb
    mask_positions : (B, G) bool
    valid_mask     : (B, G) bool  (~padding)
    """
    v          = valid_mask[0]
    tgt_all    = target[0][v]            # (V, D)
    rec_all    = recon[0][v]             # (V, D)
    m_local    = mask_positions[0][v]    # (V,)
    if tgt_all.shape[0] < 2 or m_local.sum() < 1:
        return torch.tensor(0.0, device=recon.device)
    rec_n  = F.normalize(rec_all[m_local].float(), dim=-1)   # (M, D)
    tgt_n  = F.normalize(tgt_all.float(), dim=-1)            # (V, D)
    logits = rec_n @ tgt_n.t() / temp                        # (M, V)
    labels = torch.where(m_local)[0]                         # (M,) column index of own target
    return F.cross_entropy(logits, labels)


def compute_or_load_emb_stats(cds_paths, cache_path, max_genomes=300):
    """train cds_emb 들로 차원별 mean/std 추정 (1회 계산 후 캐시).
    ★
      - R2: 캐시에 train-split 서명(sig)을 저장하고, 현재 split 과 다르면 재계산
            (옛 split / val 누수된 통계의 무단 재사용 방지).
      - R4: 분산이 거의 0 인 차원의 표준화 ÷폭발(×10^4) 방지 — std 상대 floor.
    """
    import hashlib
    sig_src = "|".join(sorted(
        os.path.basename(os.path.dirname(p)) for p in cds_paths
    ))
    sig = hashlib.md5(
        f"{sig_src}|n={len(cds_paths)}|max={max_genomes}|seed={SEED}".encode()
    ).hexdigest()

    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cached_sig = str(d["sig"]) if "sig" in d.files else None
        if cached_sig == sig:
            print(f"Loaded emb stats: {cache_path}")
            return d["mean"].astype(np.float32), d["std"].astype(np.float32)
        print("[warn] emb stats 캐시가 현재 train-split 과 불일치 → 재계산")

    print(f"Computing gene_emb stats from up to {max_genomes} train genomes...")
    rng = np.random.RandomState(SEED)
    if len(cds_paths) > max_genomes:
        sel = [cds_paths[i] for i in rng.choice(len(cds_paths), max_genomes, replace=False)]
    else:
        sel = cds_paths

    n = 0
    s = np.zeros(EMB_DIM, dtype=np.float64)
    ss = np.zeros(EMB_DIM, dtype=np.float64)
    for p in tqdm(sel, desc="emb stats"):
        e = np.load(p).astype(np.float64)
        if e.size == 0:
            continue
        s  += e.sum(0)
        ss += (e ** 2).sum(0)
        n  += e.shape[0]

    mean = (s / n)
    var  = ss / n - mean ** 2
    std  = np.sqrt(np.clip(var, 0.0, None))
    # ★ R4: 거의 상수인 차원의 ÷폭발 방지 (상대 floor = 0.1 × median std)
    pos_std = std[std > 0]
    med     = float(np.median(pos_std)) if pos_std.size else 1.0
    floor   = max(1e-3, 0.1 * med)
    std     = np.maximum(std, floor)

    mean = mean.astype(np.float32)
    std  = std.astype(np.float32)
    np.savez(cache_path, mean=mean, std=std, sig=sig)
    print(f"emb stats cached → {cache_path}  (n_genes={n:,}, std_floor={floor:.4g})")
    return mean, std


def load_cluster_map(path):
    """mash_clusters.tsv → {GCF_accession: cluster_id}."""
    m = {}
    if not os.path.exists(path):
        print(f"[warn] mash clusters not found: {path} (fallback: per-genome split)")
        return m
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)  # header
        for row in r:
            if len(row) < 2:
                continue
            gid = os.path.basename(row[0])
            for ext in (".fna", ".fasta", ".fa"):
                gid = gid.replace(ext, "")
            m[gid] = row[1]
    return m


def group_split_indices(dataset, mash_path, val_frac=0.1, seed=SEED):
    """mash 클러스터 단위 train/val split (계통학적 누수 방지)."""
    cmap   = load_cluster_map(mash_path)
    groups = []
    for cds_path, _ in dataset.samples:
        gid = os.path.basename(os.path.dirname(cds_path))
        groups.append(cmap.get(gid, f"__self_{gid}"))

    uniq = sorted(set(groups))
    rng  = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val        = max(1, int(len(uniq) * val_frac))
    val_clusters = set(uniq[:n_val])

    train_idx = [i for i, g in enumerate(groups) if g not in val_clusters]
    val_idx   = [i for i, g in enumerate(groups) if g in val_clusters]
    print(f"Split by mash cluster: {len(uniq)} clusters → "
          f"train {len(train_idx)} / val {len(val_idx)} genomes "
          f"({len(val_clusters)} val clusters)")
    return train_idx, val_idx


# ==========================================
# 4. Longformer-style Genome Attention
# ==========================================
class LongformerGenomeAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, local_window: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model      = d_model
        self.nhead        = nhead
        self.head_dim     = d_model // nhead
        self.local_window = local_window
        self.scale        = self.head_dim ** -0.5
        self.q   = nn.Linear(d_model, d_model)
        self.k   = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_mask, key_padding_mask=None, precomp=None, rel_bias=None):
        # precomp: optional topology-aware local window {col_idx_c (N,span), valid_col (N,span)}.
        #          If None -> default linear clamp window (default linear behavior).
        # rel_bias: optional (H, span) learned relative-position bias added to local scores.
        B, N, D = x.shape
        H, W, d = self.nhead, self.local_window, self.head_dim

        Q = self.q(x).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.k(x).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.v(x).view(B, N, H, d).permute(0, 2, 1, 3)

        INF      = 1e4
        pad_bias = key_padding_mask.to(x.dtype) * -INF \
                   if key_padding_mask is not None \
                   else torch.zeros(B, N, device=x.device, dtype=x.dtype)

        span = 2 * W + 1
        if precomp is not None:
            col_idx_c = precomp["col_idx_c"]        # (N, span) topology-aware (wrap/clamp, contig-bounded)
            valid_col = precomp["valid_col"]        # (N, span)
        else:
            row_idx    = torch.arange(N, device=x.device)
            col_offset = torch.arange(-W, W + 1, device=x.device)
            col_idx    = row_idx.unsqueeze(1) + col_offset.unsqueeze(0)
            valid_col  = (col_idx >= 0) & (col_idx < N)
            col_idx_c  = col_idx.clamp(0, N - 1)

        ROW_CHUNK    = 512
        loc_out_list = []
        loc_lse_list = []

        for r_start in range(0, N, ROW_CHUNK):
            r_end  = min(r_start + ROW_CHUNK, N)
            rc     = r_end - r_start
            cidx_r = col_idx_c[r_start:r_end]
            valid_r= valid_col[r_start:r_end]

            cidx_flat = cidx_r.reshape(1, 1, rc * span).expand(B, H, -1)
            cidx_d    = cidx_flat.unsqueeze(-1).expand(-1, -1, -1, d)

            K_win_r = torch.gather(K, 2, cidx_d).view(B, H, rc, span, d)
            V_win_r = torch.gather(V, 2, cidx_d).view(B, H, rc, span, d)
            Q_r     = Q[:, :, r_start:r_end, :]

            s_r = (Q_r.unsqueeze(3) * K_win_r).sum(-1) * self.scale
            if rel_bias is not None:                       # (H, span) -> (1, H, 1, span)
                s_r = s_r + rel_bias.view(1, H, 1, span)
            s_r = s_r.masked_fill(
                ~valid_r.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1), -INF
            )
            pad_r = torch.gather(
                pad_bias, 1,
                cidx_r.reshape(1, rc * span).expand(B, -1)
            )
            s_r = s_r + pad_r.view(B, 1, rc, span)

            s_r = s_r.float()                                  # ★ R3: softmax/lse 는 fp32
            loc_lse_list.append(torch.logsumexp(s_r, dim=-1))
            a_r = self.dropout(torch.softmax(s_r, dim=-1)).to(V_win_r.dtype)
            loc_out_list.append((a_r.unsqueeze(-1) * V_win_r).sum(-2))

        loc_out = torch.cat(loc_out_list, dim=2)
        loc_lse = torch.cat(loc_lse_list, dim=2)

        g_idx   = global_mask[0].nonzero(as_tuple=True)[0]
        K_g_num = g_idx.numel()

        if K_g_num > 0:
            Q_g    = Q[:, :, g_idx, :]
            gs     = torch.matmul(Q_g, K.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                gs = gs + pad_bias.unsqueeze(1).unsqueeze(2)
            gs     = gs.float()                                # ★ R3
            g_attn = self.dropout(torch.softmax(gs, dim=-1)).to(V.dtype)
            g_out  = torch.matmul(g_attn, V)
            g_lse  = torch.logsumexp(gs, dim=-1)

            K_g_keys = K[:, :, g_idx, :]
            V_g_keys = V[:, :, g_idx, :]
            ng_out_list = []
            ng_lse_list = []

            for r_start in range(0, N, ROW_CHUNK):
                r_end = min(r_start + ROW_CHUNK, N)
                Q_r   = Q[:, :, r_start:r_end, :]
                ngs_r = torch.matmul(Q_r, K_g_keys.transpose(-2, -1)) * self.scale
                if key_padding_mask is not None:
                    g_pad = key_padding_mask[:, g_idx].to(x.dtype) * -INF
                    ngs_r = ngs_r + g_pad.unsqueeze(1).unsqueeze(2)
                ngs_r = ngs_r.float()                          # ★ R3
                ng_lse_list.append(torch.logsumexp(ngs_r, dim=-1))
                ng_attn_r = self.dropout(torch.softmax(ngs_r, dim=-1)).to(V_g_keys.dtype)
                ng_out_list.append(torch.matmul(ng_attn_r, V_g_keys))

            ng_out = torch.cat(ng_out_list, dim=2)
            ng_lse = torch.cat(ng_lse_list, dim=2)

            lse_max  = torch.maximum(loc_lse, ng_lse)
            w_loc    = torch.exp(loc_lse - lse_max)
            w_ng     = torch.exp(ng_lse  - lse_max)
            denom    = (w_loc + w_ng).unsqueeze(-1).clamp(min=1e-6)
            combined = (loc_out * w_loc.unsqueeze(-1) +
                        ng_out  * w_ng.unsqueeze(-1)) / denom

            idx_exp   = g_idx.view(1, 1, K_g_num, 1).expand(B, H, -1, d)
            final_out = combined.scatter(2, idx_exp, g_out.to(combined.dtype))
        else:
            final_out = loc_out

        out = final_out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.out(out)


class LongformerGenomeLayer(nn.Module):
    def __init__(self, d_model, nhead, local_window,
                 dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = LongformerGenomeAttention(d_model, nhead, local_window, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, global_mask, key_padding_mask=None, precomp=None, rel_bias=None):
        x = x + self.attn(self.norm1(x), global_mask, key_padding_mask, precomp, rel_bias)
        x = x + self.ff(self.norm2(x))
        return x


class LongformerGenomeTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, local_window, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LongformerGenomeLayer(d_model, nhead, local_window, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, global_mask, key_padding_mask=None, precomp=None, rel_bias=None):
        from torch.utils.checkpoint import checkpoint

        kpm = key_padding_mask if key_padding_mask is not None else \
              torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        def make_layer_fn(layer):
            def fn(x, gm, kpm): return layer(x, gm, kpm, precomp, rel_bias)
            return fn

        for layer in self.layers:
            x = checkpoint(make_layer_fn(layer), x, global_mask, kpm,
                           use_reentrant=False)
        return self.norm(x)


# ==========================================
# 5. Promoter Encoder  (attention pooling)
# ==========================================
class PromoterEncoder(nn.Module):
    def __init__(self, in_dim=5, d_model=256, max_len=300, conv_dim=128):
        super().__init__()
        # ★ conv 를 좁은 폭(conv_dim)에서 수행 후 d_model 로 projection.
        #   DNA one-hot(채널 5) 처리에 d_model(480)채널 conv 는 과함 →
        #   conv FLOPs 를 (d_model/conv_dim)^2 ≈ 14배 절감 (학습 병목 해소).
        C = conv_dim
        self.input_proj = nn.Linear(in_dim, C)
        self.conv3  = nn.Conv1d(C, C, kernel_size=3,  padding=1)
        self.conv7  = nn.Conv1d(C, C, kernel_size=7,  padding=3)
        self.conv15 = nn.Conv1d(C, C, kernel_size=15, padding=7)
        self.conv31 = nn.Conv1d(C, C, kernel_size=31, padding=15)
        self.norm   = nn.LayerNorm(C)
        # 위치 보존 attention pooling (TSS 근접 모티프의 '위치' 보존 → 발현 신호)
        self.attn_query = nn.Parameter(torch.randn(C) * (C ** -0.5))
        self.scale      = C ** -0.5
        self.out_proj   = nn.Linear(C, d_model)            # conv_dim → d_model

    def forward(self, x, valid_mask):
        B, O, L, Cin = x.shape
        D = self.out_proj.out_features

        vm_flat = valid_mask.reshape(B * O)
        x_flat  = x.reshape(B * O, L, Cin)
        x_valid = x_flat[vm_flat]

        h  = self.input_proj(x_valid)
        hc = h.transpose(1, 2)
        h  = (h + self.conv3(hc).transpose(1, 2)
                + self.conv7(hc).transpose(1, 2)
                + self.conv15(hc).transpose(1, 2)
                + self.conv31(hc).transpose(1, 2)) / 5.0
        h  = self.norm(h)                                  # (Nv, L, conv_dim)

        scores    = (h @ self.attn_query) * self.scale     # (Nv, L)
        w         = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled    = (h * w).sum(dim=1)                      # (Nv, conv_dim)
        out_valid = self.out_proj(pooled)                  # (Nv, D)

        out = torch.zeros(B * O, D, device=x.device, dtype=out_valid.dtype)
        out = out.scatter(
            0,
            vm_flat.nonzero(as_tuple=True)[0].unsqueeze(-1).expand(-1, D),
            out_valid,
        )
        return out.reshape(B, O, D)


# ==========================================
# 6. Promoter Numeric Encoder
# ==========================================
class PromoterFeatureEncoder(nn.Module):
    def __init__(self, in_dim=4, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
    def forward(self, x):
        return self.net(x)


# ==========================================
# 7. Gene Token Encoder  (★ role 제거)
# ==========================================
class GeneTokenEncoder(nn.Module):
    def __init__(self, emb_dim=480, d_model=480, num_strand_types=3):
        super().__init__()
        d_strand  = d_model // 4
        d_spacing = d_model // 4

        self.gene_proj    = nn.Linear(emb_dim, d_model)
        self.strand_emb   = nn.Embedding(num_strand_types, d_strand)
        self.spacing_proj = nn.Sequential(
            nn.Linear(1, d_spacing), nn.GELU(),
            nn.Linear(d_spacing, d_spacing),
        )
        concat_dim = d_model + d_strand + d_spacing
        self.fuse = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, gene_emb, strand_idx, intergenic_spacing):
        h_gene    = self.gene_proj(gene_emb)
        h_strand  = self.strand_emb(strand_idx)
        h_spacing = self.spacing_proj(intergenic_spacing.unsqueeze(-1))
        return self.fuse(torch.cat([h_gene, h_strand, h_spacing], dim=-1))


# ==========================================
# 8. Operon Encoder
# ==========================================
class OperonEncoder(nn.Module):
    def __init__(self, d_model=480, nhead=8, num_layers=1,
                 max_genes_per_operon=24):
        super().__init__()
        self.W = max_genes_per_operon
        # ★ 위치별 분화를 빠르게 하기 위해 zeros 대신 소량 random 초기화
        self.gene_pos_emb = nn.Parameter(
            torch.randn(1, max_genes_per_operon, d_model) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_tokens_flat, gene_mask_flat):
        """
        반환:
          out    (N_ops, D)      operon 단위 pooled 표현 (genome transformer 입력용)
          h_slot (N_ops, W, D)   ★ pooling 전 per-gene(slot) 표현.
                                 gene_pos_emb 덕분에 operon 내 위치별로 구별됨
                                 → masked operon 복원에 사용(R1 비퇴화).
        """
        N_ops, W, D = gene_tokens_flat.shape
        h = gene_tokens_flat + self.gene_pos_emb[:, :W, :]

        safe_mask  = gene_mask_flat.clone()
        empty_rows = safe_mask.all(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = False
            h[empty_rows, 0] = 0.0

        h   = self.transformer(h, src_key_padding_mask=safe_mask)
        h   = self.norm(h)
        out = masked_mean(h, ~safe_mask, dim=1)
        if empty_rows.any():
            out[empty_rows] = 0.0
        return out, h


# ==========================================
# 9. Global Context Encoder
# ==========================================
class GlobalContextEncoder(nn.Module):
    def __init__(self, input_dim=12, d_model=480):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),   nn.GELU(), nn.LayerNorm(d_model),
        )
    def forward(self, x):
        return self.net(x)


# ==========================================
# 10. Hierarchical Genome BERT
# ==========================================
class HierarchicalGenomeBERT(nn.Module):
    def __init__(
        self,
        gene_emb_dim         = 480,
        global_input_dim     = 12,
        promoter_numeric_dim = 4,
        d_model              = 480,
        nhead                = 8,
        genome_layers        = 2,
        max_operons          = 9000,
        local_window         = 64,
        max_size_global      = 64,
        promoter_len         = 300,
        max_genes_per_operon = 24,
        topology_aware       = False,   # ★ circular/linear-aware genome attention + hierarchical global
        n_registers          = 16,      # ★ learned global "register" tokens (replace largest-operon heuristic)
    ):
        super().__init__()
        self.W               = max_genes_per_operon
        self.d_model         = d_model
        self.max_size_global = max_size_global
        self.topology_aware  = topology_aware
        self.nhead           = nhead
        self.local_window    = local_window
        self.n_registers     = n_registers

        # ★ 임베딩 표준화 공간 → mask_token 을 평균(=0)으로 초기화
        self.mask_token = nn.Parameter(torch.zeros(1, 1, gene_emb_dim))
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))

        self.gene_token_encoder    = GeneTokenEncoder(
            emb_dim=gene_emb_dim, d_model=d_model,
            num_strand_types=NUM_STRAND_TYPES,
        )
        self.operon_encoder        = OperonEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=1, max_genes_per_operon=max_genes_per_operon,
        )
        self.promoter_encoder      = PromoterEncoder(
            in_dim=5, d_model=d_model, max_len=promoter_len,
        )
        self.promoter_feat_encoder = PromoterFeatureEncoder(
            in_dim=promoter_numeric_dim, d_model=d_model,
        )
        self.global_encoder        = GlobalContextEncoder(
            input_dim=global_input_dim, d_model=d_model,
        )
        self.operon_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.genome_pos_emb = nn.Parameter(
            torch.zeros(1, max_operons + 2, d_model)
        )
        self.genome_transformer = LongformerGenomeTransformer(
            d_model=d_model, nhead=nhead,
            num_layers=genome_layers, local_window=local_window,
        )
        self.gene_reconstruct_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, gene_emb_dim),
        )
        self.bgc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        nn.init.zeros_(self.bgc_head[-1].weight)
        nn.init.zeros_(self.bgc_head[-1].bias)

        # ★ topology-aware genome attention + hierarchical global tokens
        if topology_aware:
            # (b) linear/circular embedding, (d) distance-from-end (linear arms) — zero-init
            self.topology_emb = nn.Embedding(2, d_model)          # 0=linear, 1=circular
            nn.init.zeros_(self.topology_emb.weight)
            self.enddist_proj = nn.Linear(1, d_model)
            nn.init.zeros_(self.enddist_proj.weight); nn.init.zeros_(self.enddist_proj.bias)
            # (c) relative-position bias in the local window (replaces absolute genome_pos_emb)
            self.rel_pos_bias = nn.Parameter(torch.zeros(nhead, 2 * local_window + 1))
            # (A) learned register tokens  +  (B) contig-summary projection
            self.register_tokens = nn.Parameter(torch.randn(1, n_registers, d_model) * 0.02)
            self.contig_proj     = nn.Linear(d_model, d_model)

    def _build_topo_window(self, N, P, O, cs, cl, tp, dev):
        """Per-token local-window column indices (N, span) that (i) stay inside each
        operon's contig block, (ii) WRAP for circular contigs / clamp for linear,
        (iii) give prefix tokens [cls,global,registers,contig] a self-only window
        (they use the global path). cs/cl/tp are per-operon (O,)."""
        W = self.local_window; span = 2 * W + 1
        tok_start = torch.arange(N, device=dev)                     # prefix -> self
        tok_len   = torch.ones(N, dtype=torch.long, device=dev)
        tok_circ  = torch.zeros(N, dtype=torch.bool, device=dev)
        tok_start[P:P + O] = P + cs                                 # operon block start in token space
        tok_len[P:P + O]   = cl
        tok_circ[P:P + O]  = (tp == 1)
        row = torch.arange(N, device=dev)
        pos = row - tok_start                                       # position within block
        off = torch.arange(-W, W + 1, device=dev)
        npc = pos.unsqueeze(1) + off.unsqueeze(0)                   # (N, span) neighbor pos-in-block
        L   = tok_len.unsqueeze(1)
        circ = tok_circ.unsqueeze(1)
        npc_f = torch.where(circ, npc % L, npc)                     # wrap (circular) / raw (linear)
        valid = torch.where(circ, torch.ones_like(npc, dtype=torch.bool),
                            (npc >= 0) & (npc < L))                 # circular always valid; linear in-range
        col_idx   = tok_start.unsqueeze(1) + npc_f
        valid     = valid & (col_idx >= 0) & (col_idx < N)
        col_idx_c = col_idx.clamp(0, N - 1)
        return {"col_idx_c": col_idx_c, "valid_col": valid}

    def _genome_forward(self, fused_operon, cls_token, global_token, operon_valid_mask,
                        operon_sizes, operon_contig_start, operon_contig_len, operon_topology):
        """Run the genome transformer. base path (absolute pos emb + largest-operon global) when
        topology info is absent; topology-aware + hierarchical-global path when present.
        Returns (cls_out, global_out, operon_ctx) with operon_ctx in operon order (B,O,D)."""
        B, O, D = fused_operon.shape
        dev = fused_operon.device
        use_topo = self.topology_aware and (operon_contig_start is not None)

        if not use_topo:
            genome_tokens = torch.cat([cls_token, global_token, fused_operon], dim=1)
            N = genome_tokens.size(1)
            genome_tokens = genome_tokens + self.genome_pos_emb[:, :N, :]
            genome_padding_mask = torch.cat(
                [torch.zeros(B, 2, device=dev, dtype=torch.bool), ~operon_valid_mask], dim=1)
            global_mask = self._build_global_mask(B, N, operon_valid_mask, operon_sizes, dev)
            genome_out = self.genome_transformer(
                genome_tokens, global_mask=global_mask, key_padding_mask=genome_padding_mask)
            return genome_out[:, 0, :], genome_out[:, 1, :], genome_out[:, 2:, :]

        # ---- topology-aware + hierarchical-global ----
        cs = operon_contig_start[0].long()
        cl = operon_contig_len[0].long().clamp(min=1)
        tp = operon_topology[0].long()
        ov = operon_valid_mask[0].float().unsqueeze(-1)

        # (b) topology emb + (d) distance-from-end, added to operon tokens
        op_pos   = torch.arange(O, device=dev) - cs
        dfe      = torch.minimum(op_pos, cl - 1 - op_pos).float()
        dfe_norm = (2.0 * dfe / cl.float()).clamp(0, 1)            # 0 at end, 1 at centre (linear)
        dfe_norm = torch.where(tp == 1, torch.ones_like(dfe_norm), dfe_norm)  # circular: no ends
        op = fused_operon + self.topology_emb(tp).unsqueeze(0) \
                          + self.enddist_proj(dfe_norm.view(1, O, 1))

        # (B) contig-summary tokens: mean-pool valid operons per contig block -> proj
        uniq_starts, seg = torch.unique(cs, return_inverse=True)   # M contigs, seg:(O,)
        M    = uniq_starts.numel()
        csum = torch.zeros(M, D, device=dev).index_add_(0, seg, fused_operon[0] * ov)
        ccnt = torch.zeros(M, 1, device=dev).index_add_(0, seg, ov).clamp(min=1)
        contig_tok = self.contig_proj(csum / ccnt).unsqueeze(0)    # (1, M, D)

        # (A) learned register tokens
        reg = self.register_tokens.expand(B, -1, -1)               # (1, K, D)
        prefix = torch.cat([cls_token, global_token, reg, contig_tok], dim=1)
        P = prefix.size(1)                                         # 2 + K + M
        genome_tokens = torch.cat([prefix, op], dim=1)
        N = genome_tokens.size(1)

        genome_padding_mask = torch.cat(
            [torch.zeros(B, P, device=dev, dtype=torch.bool), ~operon_valid_mask], dim=1)
        global_mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        global_mask[:, :P] = True                                  # all prefix tokens are global

        precomp = self._build_topo_window(N, P, O, cs, cl, tp, dev)
        genome_out = self.genome_transformer(
            genome_tokens, global_mask=global_mask, key_padding_mask=genome_padding_mask,
            precomp=precomp, rel_bias=self.rel_pos_bias)
        return genome_out[:, 0, :], genome_out[:, 1, :], genome_out[:, P:, :]

    def _build_operon_tokens(self, gene_tokens, op_ids, op_valid, O):
        """
        ★ 벡터화.
          - n_genes ≤ W 오페론: scatter 로 완전 벡터화.
          - n_genes > W 오페론(희소): 기존 sliding-window 평균 루프 유지.
        """
        W, D   = self.W, gene_tokens.shape[-1]
        device = gene_tokens.device
        Gv     = gene_tokens.shape[0]

        tokens = torch.zeros(O, W, D, device=device, dtype=gene_tokens.dtype)
        masks  = torch.ones(O, W,    device=device, dtype=torch.bool)
        if Gv == 0 or O == 0:
            return tokens, masks, torch.zeros(Gv, dtype=torch.long, device=device)

        counts      = torch.bincount(op_ids, minlength=O)            # (O,)
        order       = torch.argsort(op_ids, stable=True)
        sorted_ops  = op_ids[order]
        group_start = torch.cumsum(counts, 0) - counts
        intra       = torch.arange(Gv, device=device) - group_start[sorted_ops]

        # ★ 각 (유효)유전자의 operon 내 slot 위치 (gene 순서로 복원)
        slot_of_gene        = torch.zeros(Gv, dtype=torch.long, device=device)
        slot_of_gene[order] = intra

        op_size_of_gene = counts[sorted_ops]
        sel = (intra < W) & (op_size_of_gene <= W)
        g_sel = order[sel]
        o_sel = sorted_ops[sel]
        r_sel = intra[sel]
        tokens[o_sel, r_sel] = gene_tokens[g_sel]
        masks[o_sel, r_sel]  = False

        # n_genes > W 오페론 (희소): 기존 동작 그대로 (sliding-window 평균)
        big_ops = torch.where(counts > W)[0]
        for oi in big_ops.tolist():
            if not bool(op_valid[oi]):
                continue
            sel2 = (op_ids == oi).nonzero(as_tuple=True)[0]
            g_seq = gene_tokens[sel2]
            n_g   = g_seq.shape[0]
            starts = list(range(0, n_g - W + 1, OPERON_WINDOW_STRIDE))
            if starts[-1] != n_g - W:
                starts.append(n_g - W)
            wins = torch.stack([g_seq[s:s + W] for s in starts])
            tokens[oi] = wins.mean(dim=0)
            masks[oi]  = False

        return tokens, masks, slot_of_gene

    def _build_global_mask(self, B, N, operon_valid_mask, operon_sizes, device):
        """
        ★ global 토큰을 '가장 큰 오페론'에만 두지 않고,
          (1) 절반 = 큰 오페론(기능 허브 후보)
          (2) 나머지 = genome 전반에 균등 분포한 long-range 앵커
          로 분산 → 64-오페론 local window 밖의 whole-genome 맥락 전달 개선.
        cls(0) / global-feature(1) 토큰은 항상 global.
        """
        global_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        global_mask[:, 0] = True   # cls
        global_mask[:, 1] = True   # global-feature token

        num_valid = operon_valid_mask.sum(dim=1)
        k = min(self.max_size_global, int(num_valid.max().item()))
        if k <= 0:
            return global_mask

        sizes = operon_sizes.float() * operon_valid_mask.float()
        for b in range(B):
            nv = int(num_valid[b].item())
            if nv == 0:
                continue
            kb = min(k, nv)

            chosen = set()
            # (1) 가장 큰 오페론 (기능적 허브 후보)
            k_big = kb // 2
            if k_big > 0:
                chosen.update(sizes[b].topk(k_big).indices.tolist())

            # (2) genome 전반에 균등 분포한 long-range 앵커
            valid_pos = torch.where(operon_valid_mask[b])[0]
            k_even = kb - len(chosen)
            if k_even > 0 and valid_pos.numel() > 0:
                sel = torch.linspace(0, valid_pos.numel() - 1, k_even,
                                     device=device).round().long()
                chosen.update(valid_pos[sel].tolist())

            if chosen:
                pos = (torch.tensor(sorted(chosen), device=device) + 2).clamp(max=N - 1)
                global_mask[b, pos] = True

        return global_mask

    def forward(
        self,
        gene_emb,
        gene_padding_mask,
        operon_id_of_gene,
        strand_idx,
        intergenic_spacing,
        promoter_onehot,
        promoter_numeric_features,
        operon_valid_mask,
        operon_sizes,
        global_features,
        masked_gene_positions=None,
        operon_contig_start=None,   # ★ topology: per-operon contig block start (O,)
        operon_contig_len=None,     # ★ topology: per-operon contig length (O,)
        operon_topology=None,       # ★ topology: per-operon 1=circular / 0=linear (O,)
    ):
        B, G, _ = gene_emb.shape
        assert B == 1, "forward 는 batch_size=1 전제 (operon 처리/global_mask 가 [0] 인덱싱)"
        _, O, _, _ = promoter_onehot.shape
        D = self.d_model

        if masked_gene_positions is not None:
            gene_emb = gene_emb.clone()
            gene_emb[masked_gene_positions] = self.mask_token[0, 0].to(gene_emb.dtype)

        gene_tokens = self.gene_token_encoder(
            gene_emb, strand_idx, intergenic_spacing
        )

        valid_gene   = ~gene_padding_mask[0]
        gt_valid     = gene_tokens[0][valid_gene]
        op_ids_valid = operon_id_of_gene[0][valid_gene]

        op_tokens, op_masks, slot_of_gene = self._build_operon_tokens(
            gt_valid, op_ids_valid, operon_valid_mask[0], O
        )
        # 대형 operon(>W) 유전자는 slot 을 W-1 로 clamp (window-mean 출력으로 fallback)
        slot_clamped = slot_of_gene.clamp(max=self.W - 1)

        OPERON_CHUNK = 256
        operon_repr_chunks = []
        # ★ pooling 전 per-gene(slot) 표현을 청크별로 즉시 gather
        #   → (O,W,D) 전체를 materialize 하지 않고 (Gv,D) 만 유지 (메모리 절감)
        Gv = gt_valid.shape[0]
        gene_in_op_valid = torch.zeros(Gv, self.d_model,
                                       device=gt_valid.device, dtype=gt_valid.dtype)
        for s in range(0, O, OPERON_CHUNK):
            e = min(s + OPERON_CHUNK, O)
            pooled, h_slot = self.operon_encoder(op_tokens[s:e], op_masks[s:e])
            operon_repr_chunks.append(pooled)
            in_chunk = (op_ids_valid >= s) & (op_ids_valid < e)
            if in_chunk.any():
                gi        = in_chunk.nonzero(as_tuple=True)[0]
                local_op  = op_ids_valid[gi] - s
                local_slt = slot_clamped[gi]
                gene_in_op_valid[gi] = h_slot[local_op, local_slt]
        operon_repr = (torch.cat(operon_repr_chunks, dim=0).unsqueeze(0)
                       if operon_repr_chunks else
                       torch.zeros(1, 0, self.d_model, device=gt_valid.device))

        # 유효 유전자 → 전체 (B,G) 위치로 복원 (per-gene, operon 내 위치-aware)
        gene_in_operon = torch.zeros(B, G, self.d_model,
                                     device=gene_emb.device,
                                     dtype=gene_in_op_valid.dtype)
        gene_in_operon[0][valid_gene] = gene_in_op_valid

        promoter_seq_repr = self.promoter_encoder(promoter_onehot, operon_valid_mask)
        promoter_num_repr = self.promoter_feat_encoder(promoter_numeric_features)

        valid_float       = operon_valid_mask.float().unsqueeze(-1)
        operon_repr       = operon_repr       * valid_float
        promoter_seq_repr = promoter_seq_repr * valid_float
        promoter_num_repr = promoter_num_repr * valid_float

        fused_operon = self.operon_fuse(
            torch.cat([operon_repr, promoter_seq_repr, promoter_num_repr], dim=-1)
        )

        global_token  = self.global_encoder(global_features).unsqueeze(1)
        cls_token     = self.cls_token.expand(B, -1, -1)
        cls_out, global_out, operon_ctx = self._genome_forward(
            fused_operon, cls_token, global_token, operon_valid_mask, operon_sizes,
            operon_contig_start, operon_contig_len, operon_topology,
        )

        if O > 0:
            gather_idx = (
                operon_id_of_gene.clamp(0, O - 1)
                .unsqueeze(-1).expand(-1, -1, D)
            )
            gene_operon_ctx = torch.gather(operon_ctx, dim=1, index=gather_idx)
        else:
            gene_operon_ctx = torch.zeros_like(gene_tokens)

        # ★ 첫 항을 gene_tokens(masked, operon 내 동일)에서
        #   gene_in_operon(operon 내 위치-aware per-gene)으로 교체
        #   → masked operon 복원이 위치별로 구별되어 R1(붕괴) 해소
        recon = self.gene_reconstruct_head(
            torch.cat([gene_in_operon, gene_operon_ctx], dim=-1)
        )

        bgc_pred = self.bgc_head(operon_ctx)  # (B, O, 3)

        return {
            "recon_gene":      recon,
            "cls":             cls_out,
            "global":          global_out,
            "operon_ctx":      operon_ctx,
            "gene_operon_ctx": gene_operon_ctx,
            "gene_tokens":     gene_tokens,
            "gene_in_operon":  gene_in_operon,
            "bgc_pred":        bgc_pred,
        }


# ==========================================
# 11. Dataset  (★ role 제거 + 표준화)
# ==========================================
class HierGenomeDataset(Dataset):
    def __init__(self, root_dir, max_genes=8092, max_operons=9000,
                 promoter_len=300):
        self.samples = []
        for gd in sorted(glob.glob(os.path.join(root_dir, "*"))):
            cds_path  = os.path.join(gd, "cds_emb.npy")
            meta_path = os.path.join(gd, "operon_meta.npz")
            if os.path.exists(cds_path) and os.path.exists(meta_path):
                self.samples.append((cds_path, meta_path))
        self.max_genes    = max_genes
        self.max_operons  = max_operons
        self.promoter_len = promoter_len
        # ★ 표준화 통계 (train 으로 계산해 외부에서 주입)
        self.emb_mean = None
        self.emb_std  = None

    def set_emb_stats(self, mean, std):
        self.emb_mean = torch.from_numpy(np.asarray(mean, dtype=np.float32))
        self.emb_std  = torch.from_numpy(np.asarray(std,  dtype=np.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cds_path, meta_path = self.samples[idx]
        gene_emb = np.load(cds_path).astype(np.float32)
        meta     = np.load(meta_path, allow_pickle=True)

        operon_id_of_gene         = meta["operon_id_of_gene"].astype(np.int64)
        gene_strand               = meta["gene_strand"].astype(np.int64)
        intergenic_upstream       = meta["intergenic_upstream"].astype(np.float32)
        promoter_onehot           = meta["promoter_onehot"].astype(np.float32)
        promoter_numeric_features = meta["promoter_numeric_features"].astype(np.float32)
        operon_valid_mask         = meta["operon_valid_mask"].astype(bool)
        operon_sizes              = meta["operon_sizes"].astype(np.int64)
        global_features           = meta["global_features"].astype(np.float32)
        operon_bgc_score          = meta["operon_bgc_score"].astype(np.float32)
        operon_bgc_mask           = meta["operon_bgc_mask"].astype(bool)

        promoter_numeric_features = promoter_numeric_features[:, :PROMOTER_NUMERIC_DIM]

        n = min(len(gene_emb), len(operon_id_of_gene),
                len(gene_strand), len(intergenic_upstream))

        gene_emb            = gene_emb[:n]
        operon_id_of_gene   = operon_id_of_gene[:n]
        gene_strand         = gene_strand[:n]
        intergenic_upstream = intergenic_upstream[:n]

        gene_strand         = np.where(gene_strand < 0, 1, 2).astype(np.int64)
        intergenic_upstream = np.log1p(
            np.clip(intergenic_upstream, 0, 5000)
        ).astype(np.float32)

        O = min(len(promoter_onehot), self.max_operons)
        promoter_onehot           = promoter_onehot[:O, :self.promoter_len, :]
        promoter_numeric_features = promoter_numeric_features[:O]
        operon_valid_mask         = operon_valid_mask[:O]
        operon_sizes              = operon_sizes[:O]
        operon_bgc_score          = operon_bgc_score[:O]
        operon_bgc_mask           = operon_bgc_mask[:O]

        # ★ per-operon topology (contig block start/len + linear/circular)
        topo_path = os.path.join(os.path.dirname(meta_path), "topology_meta.npz")
        if os.path.exists(topo_path):
            tm = np.load(topo_path)
            op_contig_start = tm["operon_contig_start"].astype(np.int64)[:O]
            op_contig_len   = tm["operon_contig_len"].astype(np.int64)[:O]
            op_topology     = tm["operon_topology"].astype(np.int64)[:O]
        else:                                   # fallback: whole genome as one circular contig
            op_contig_start = np.zeros(O, np.int64)
            op_contig_len   = np.full(O, max(O, 1), np.int64)
            op_topology     = np.ones(O, np.int64)
        if O > 0:                               # keep contig block inside the kept operon range
            op_contig_len = np.minimum(op_contig_len, O - op_contig_start).clip(min=1)

        if O > 0:
            keep              = operon_id_of_gene < O
            gene_emb          = gene_emb[keep]
            operon_id_of_gene = operon_id_of_gene[keep]
            gene_strand       = gene_strand[keep]
            intergenic_upstream = intergenic_upstream[keep]
        else:
            gene_emb          = np.zeros((0, EMB_DIM), dtype=np.float32)
            operon_id_of_gene = np.zeros((0,), dtype=np.int64)
            gene_strand       = np.zeros((0,), dtype=np.int64)
            intergenic_upstream = np.zeros((0,), dtype=np.float32)

        G = min(len(gene_emb), self.max_genes)
        gene_emb            = gene_emb[:G]
        operon_id_of_gene   = operon_id_of_gene[:G]
        gene_strand         = gene_strand[:G]
        intergenic_upstream = intergenic_upstream[:G]

        gene_emb = torch.from_numpy(gene_emb)
        # ★ 차원별 표준화 (train 통계)
        if self.emb_mean is not None and gene_emb.shape[0] > 0:
            gene_emb = (gene_emb - self.emb_mean) / self.emb_std

        return {
            "gene_emb":                  gene_emb,
            "operon_id_of_gene":         torch.from_numpy(operon_id_of_gene),
            "strand_idx":                torch.from_numpy(gene_strand),
            "intergenic_spacing":        torch.from_numpy(intergenic_upstream),
            "promoter_onehot":           torch.from_numpy(promoter_onehot),
            "promoter_numeric_features": torch.from_numpy(promoter_numeric_features),
            "operon_valid_mask":         torch.from_numpy(operon_valid_mask),
            "operon_sizes":              torch.from_numpy(operon_sizes),
            "global_features":           torch.from_numpy(global_features),
            "operon_bgc_score":          torch.from_numpy(operon_bgc_score),
            "operon_bgc_mask":           torch.from_numpy(operon_bgc_mask),
            "operon_contig_start":       torch.from_numpy(op_contig_start),
            "operon_contig_len":         torch.from_numpy(op_contig_len),
            "operon_topology":           torch.from_numpy(op_topology),
        }


def collate_fn(batch: List[Dict]):
    B             = len(batch)
    max_g         = max(x["gene_emb"].shape[0] for x in batch)
    max_o         = max(x["promoter_onehot"].shape[0] for x in batch)
    prom_len      = batch[0]["promoter_onehot"].shape[1]
    prom_feat_dim = batch[0]["promoter_numeric_features"].shape[1]
    global_dim    = batch[0]["global_features"].shape[0]

    gene_emb              = torch.zeros(B, max_g, EMB_DIM)
    gene_padding_mask     = torch.ones(B, max_g, dtype=torch.bool)
    operon_id_of_gene     = torch.zeros(B, max_g, dtype=torch.long)
    strand_idx            = torch.zeros(B, max_g, dtype=torch.long)
    intergenic_spacing    = torch.zeros(B, max_g)
    promoter_onehot       = torch.zeros(B, max_o, prom_len, 5)
    promoter_numeric_feat = torch.zeros(B, max_o, prom_feat_dim)
    operon_valid_mask     = torch.zeros(B, max_o, dtype=torch.bool)
    operon_sizes          = torch.zeros(B, max_o, dtype=torch.long)
    global_features       = torch.zeros(B, global_dim)
    operon_bgc_score      = torch.zeros(B, max_o)
    operon_bgc_mask       = torch.zeros(B, max_o, dtype=torch.bool)
    operon_contig_start   = torch.zeros(B, max_o, dtype=torch.long)
    operon_contig_len     = torch.ones(B, max_o, dtype=torch.long)
    operon_topology       = torch.zeros(B, max_o, dtype=torch.long)

    for i, x in enumerate(batch):
        g = x["gene_emb"].shape[0]
        o = x["promoter_onehot"].shape[0]
        gene_emb[i, :g]              = x["gene_emb"]
        gene_padding_mask[i, :g]     = False
        operon_id_of_gene[i, :g]     = x["operon_id_of_gene"]
        strand_idx[i, :g]            = x["strand_idx"]
        intergenic_spacing[i, :g]    = x["intergenic_spacing"]
        promoter_onehot[i, :o]       = x["promoter_onehot"]
        promoter_numeric_feat[i, :o] = x["promoter_numeric_features"]
        operon_valid_mask[i, :o]     = x["operon_valid_mask"]
        operon_sizes[i, :o]          = x["operon_sizes"]
        global_features[i]           = x["global_features"]
        operon_bgc_score[i, :o]      = x["operon_bgc_score"]
        operon_bgc_mask[i, :o]       = x["operon_bgc_mask"]
        operon_contig_start[i, :o]   = x["operon_contig_start"]
        operon_contig_len[i, :o]     = x["operon_contig_len"]
        operon_topology[i, :o]       = x["operon_topology"]

    return {
        "gene_emb":                  gene_emb,
        "gene_padding_mask":         gene_padding_mask,
        "operon_id_of_gene":         operon_id_of_gene,
        "strand_idx":                strand_idx,
        "intergenic_spacing":        intergenic_spacing,
        "promoter_onehot":           promoter_onehot,
        "promoter_numeric_features": promoter_numeric_feat,
        "operon_valid_mask":         operon_valid_mask,
        "operon_sizes":              operon_sizes,
        "global_features":           global_features,
        "operon_bgc_score":          operon_bgc_score,
        "operon_bgc_mask":           operon_bgc_mask,
        "operon_contig_start":       operon_contig_start,
        "operon_contig_len":         operon_contig_len,
        "operon_topology":           operon_topology,
    }




# ==================================================================
# nucleotide cis-encoder + build functions
# ==================================================================


class DilatedResBlock(nn.Module):
    def __init__(s,C,dil,p=0.1):
        super().__init__()
        s.conv=nn.Conv1d(C,C,kernel_size=3,padding=dil,dilation=dil)
        s.norm=nn.GroupNorm(1,C); s.drop=nn.Dropout(p)   # GroupNorm(1) = LayerNorm over channels, batch-size-agnostic
    def forward(s,x):  # (N,C,L)
        return x+s.drop(F.gelu(s.norm(s.conv(x))))

class PromoterCisEncoder(nn.Module):
    """DNA cis-grammar encoder: motif conv -> dilated residual stack (multi-scale
    regulatory context: -10/-35, UP, TFBS) -> attention pooling.  in: (B,O,L,5) out: (B,O,d_model).
    Operons are processed in chunks with gradient checkpointing to bound memory."""
    def __init__(s,in_dim=5,d_model=480,conv_dim=128,max_len=300,n_dil=4,p=0.1,chunk=2048,use_ckpt=True):
        super().__init__()
        C=conv_dim; s.chunk=chunk; s.use_ckpt=use_ckpt
        s.stem=nn.Conv1d(in_dim,C,kernel_size=15,padding=7)
        s.blocks=nn.ModuleList([DilatedResBlock(C,2**i,p) for i in range(n_dil)])  # dilations 1,2,4,8
        s.norm=nn.LayerNorm(C)
        s.attn_q=nn.Parameter(torch.randn(C)*(C**-0.5)); s.scale=C**-0.5
        s.out=nn.Sequential(nn.Linear(C,d_model),nn.GELU(),nn.LayerNorm(d_model))
    def _encode(s,xf):   # (n,L,5) -> (n,d_model)
        h=s.stem(xf.transpose(1,2))
        for blk in s.blocks: h=blk(h)
        h=s.norm(h.transpose(1,2))
        w=torch.softmax((h@s.attn_q)*s.scale,dim=1).unsqueeze(-1)
        return s.out((h*w).sum(1))
    def forward(s,x,valid_mask):
        from torch.utils.checkpoint import checkpoint
        B,O,L,Cin=x.shape; D=s.out[-1].normalized_shape[0]
        vm=valid_mask.reshape(B*O); xf=x.reshape(B*O,L,Cin)[vm]
        out=torch.zeros(B*O,D,device=x.device,dtype=x.dtype)
        if xf.shape[0]>0:
            parts=[]
            for c in range(0,xf.shape[0],s.chunk):
                cx=xf[c:c+s.chunk]
                parts.append(checkpoint(s._encode,cx,use_reentrant=False) if (s.use_ckpt and s.training) else s._encode(cx))
            ov=torch.cat(parts,0)
            out=out.scatter(0,vm.nonzero(as_tuple=True)[0].unsqueeze(-1).expand(-1,D),ov.to(out.dtype))
        return out.reshape(B,O,D)

class ActinoBERT(HierarchicalGenomeBERT):
    """ActinoBERT with the deep DNA cis-encoder (hierarchy preserved)."""
    def __init__(s,*a,promoter_conv_dim=96,promoter_dils=3,promoter_use_ckpt=False,**kw):
        super().__init__(*a,**kw)
        s.promoter_encoder=PromoterCisEncoder(in_dim=5,d_model=s.d_model,
            conv_dim=promoter_conv_dim,n_dil=promoter_dils,use_ckpt=promoter_use_ckpt)

# Compatibility name for existing imports and previously serialized objects.
GenomeBERT = ActinoBERT

def build_topo_only(global_dim,warm_start=None):
    """Phase-1 model: the hierarchy + topology(a-d) + hierarchical global(registers/contig),
    KEEPING the cheap promoter encoder (no heavy cis-encoder). ~0.17s/genome."""
    m=HierarchicalGenomeBERT(gene_emb_dim=480,global_input_dim=global_dim,promoter_numeric_dim=4,
        d_model=480,nhead=8,genome_layers=2,max_operons=MAX_OPERONS,local_window=LOCAL_WINDOW,
        max_size_global=MAX_SIZE_GLOBAL,promoter_len=PROMOTER_LEN,max_genes_per_operon=MAX_GENES_PER_OPERON,
        topology_aware=True,n_registers=16)
    if warm_start:
        st=torch.load(warm_start,map_location="cpu"); own=m.state_dict()
        use={k:v for k,v in st.items() if k in own and own[k].shape==v.shape}
        own.update(use); m.load_state_dict(own)
        print(f"[topo-only] warm-started {len(use)} tensors from {os.path.basename(warm_start)}")
    return m

def build_model(global_dim,warm_start=None,
             promoter_conv_dim=96,promoter_dils=3,promoter_use_ckpt=False):
    m=ActinoBERT(gene_emb_dim=480,global_input_dim=global_dim,promoter_numeric_dim=4,
        d_model=480,nhead=8,genome_layers=2,max_operons=MAX_OPERONS,local_window=LOCAL_WINDOW,
        max_size_global=MAX_SIZE_GLOBAL,promoter_len=PROMOTER_LEN,max_genes_per_operon=MAX_GENES_PER_OPERON,
        topology_aware=True, n_registers=16,   # ★ circular/linear + hierarchical global (registers+contig)
        promoter_conv_dim=promoter_conv_dim, promoter_dils=promoter_dils, promoter_use_ckpt=promoter_use_ckpt)
    if warm_start:
        st=torch.load(warm_start,map_location="cpu")
        # load all shape-matching tensors: from a backbone ckpt the (different-shape) promoter_encoder
        # is auto-skipped (fresh cis-grammar); from a cis-encoder ckpt it loads (pretrained cis-grammar).
        own=m.state_dict(); use={k:v for k,v in st.items() if k in own and own[k].shape==v.shape}
        prom=sum(1 for k in use if k.startswith("promoter_encoder."))
        own.update(use); m.load_state_dict(own)
        print(f"[warm] warm-started {len(use)} tensors from {os.path.basename(warm_start)} "
              f"(promoter cis-encoder tensors loaded: {prom})")
    return m
