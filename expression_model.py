"""MYClN: homology-retrieval + learned-head gene-expression predictor.

Single, self-contained module for the MYClN configuration (ActinoBERT + media): it
predicts per-(gene, growth-medium) expression by combining kNN homology transfer
from training genes with a learned FiLM-conditioned residual head, LightGBM, and a
ridge stack.  Held-out-genus (leakage-safe) evaluation.

Three stages, mirroring how the model is trained and served:

    build_cache(sets, unified_data, esm_emb, esm_order, glm_emb, glm_cls, glm_order)
        Heavy, run ONCE per split.  Builds every per-gene feature table, the kNN
        homology-retrieval index, and the per-(gene, medium) expression matrix.
    derive_bundle(cache)
        Cheap.  Retrieval-decomposes (with a same-medium expression shift) and
        assembles the head feature table.  Medium conditioning is always on.
    train_head(bundle, hp=None, seeds=8)
        8-seed FiLM residual head + LightGBM + ridge stacking; returns the
        per-test-row stacked predictions and pooled / within-condition Pearson r.

Feature channels used by this configuration:
    ESM   protein ESM2-35M homology embedding                     (480-d)
    gLM   genome-context per-gene embedding, split GIO / CTX      (480-d each)
    gLM   genome-CLS embedding                                    (480-d)
    DNA   GC fraction + codon (3-mer) composition                 (65-d)
    media growth-medium vector + same-medium retrieval shift      (159-d)

Training accepts loaded arrays; optional artifact_dir exports the frozen serving
state. Runtime prediction is implemented in serving.py. The caller passes objects to
``build_cache`` (species -> ndarray / species -> list, matching the layout of
``bert_gene_raw`` / ``bert_gene_embs`` / ``bert_cls_vectors`` / ``bert_gene_id_order``).
"""
import os, time, json, numpy as np
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device(os.environ.get("ACTEXPR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
torch.backends.cuda.matmul.allow_tf32 = True
# CPU smoke runs use float32 matmul; CUDA keeps the original fp16 retrieval.
RETR_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

# retrieval decomposition batch size; multi-(k, temperature) prior ensemble grid
RETR_BS  = 8192
MULTI_KT = [(16, 0.05), (32, 0.05), (48, 0.05), (96, 0.05)]

HP_DEFAULT = dict(hid=768, nb=4, dropout=0.1, dm=128, lr=1e-3, wd=1e-4,
                  bs=32768, epochs=120, patience=16, seeds=8, ridge_lam_scale=1e-2)


# ═══════════════════════════ helpers ═══════════════════════════
def pear(p, t):
    p = p - p.mean(); t = t - t.mean()
    return float((p * t).sum() / (p.norm() * t.norm() + 1e-8))


_LUT = np.full(256, 255, np.uint8)
for _ch, _v in zip(b"ACGT", [0, 1, 2, 3]):
    _LUT[_ch] = _v


def dnasum(seq):
    """GC fraction + 64 3-mer/codon frequencies (65-d) from a nucleotide string."""
    if not seq:
        return np.zeros(65, np.float32)
    a = np.frombuffer(seq.upper().encode("ascii", "ignore"), np.uint8)
    code = _LUT[a]; valid = code < 4
    n = max(int(valid.sum()), 1); gc = float(((code == 1) | (code == 2)).sum()) / n
    if len(code) < 3:
        return np.concatenate([[gc], np.zeros(64, np.float32)]).astype(np.float32)
    idx = code[:-2].astype(np.int32) * 16 + code[1:-1] * 4 + code[2:]
    m = valid[:-2] & valid[1:-1] & valid[2:]
    cnt = np.bincount(idx[m], minlength=64)[:64].astype(np.float32)
    ss = cnt.sum(); f = cnt / ss if ss > 0 else cnt
    return np.concatenate([[gc], f]).astype(np.float32)


def within_cond(pred, true, exp, min_g=50):
    """Mean Pearson r computed *within* each (species, medium) condition group."""
    o = np.argsort(exp, kind="stable"); g = exp[o]; p = pred[o]; t = true[o]; rs = []
    for ix in np.split(np.arange(len(g)), np.flatnonzero(np.diff(g)) + 1):
        if len(ix) < min_g:
            continue
        pg, tg = p[ix], t[ix]
        if np.std(pg) < 1e-6 or np.std(tg) < 1e-6:
            continue
        r = np.corrcoef(pg, tg)[0, 1]
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


# ═══════════════════════════ build_cache ═══════════════════════════
def build_cache(sets, unified_data, esm_emb, esm_order, glm_emb, glm_cls, glm_order, artifact_dir=None):
    """Build every per-gene feature table + the full kNN retrieval index + the
    per-(gene, medium) expression matrix ONCE for a split.  Returns a dict reused
    by ``derive_bundle``.

    Parameters
    ----------
    sets       : {"train": [species...], "val": [...], "test": [...]}
    unified_data : loaded unified expression dataset (``gene_wise_data`` +
                 ``species_metadata``).
    esm_emb    : species -> ndarray of protein ESM2-35M embeddings (480-d rows).
    esm_order  : species -> list of gene ids indexing ``esm_emb`` rows.
    glm_emb    : species -> ndarray of gLM per-gene embeddings (960-d rows; the
                 first 480 are GIO, the last 480 are CTX).
    glm_cls    : species -> genome-CLS embedding (480-d).
    glm_order  : species -> list of gene ids indexing ``glm_emb`` rows.  May be the
                 same object as ``esm_order`` when the two embedding sets share an
                 ordering (a fast path is taken in that case).
    """
    export = Path(artifact_dir) if artifact_dir is not None else None
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
    d = unified_data
    gwd = d["gene_wise_data"]; smeta = d["species_metadata"]
    order = esm_order; RAWp = esm_emb; E2E = glm_emb; CLSp = glm_cls
    g2i = {sp: {g: i for i, g in enumerate(ids)} for sp, ids in order.items()}
    # gLM ordering (identical object to g2i when the two sets share an ordering)
    g2iG = (g2i if glm_order is order else
            {sp: {g: i for i, g in enumerate(ids)} for sp, ids in glm_order.items()})
    sets = {s: set(sets[s]) for s in ["train", "val", "test"]}

    t0 = time.time(); print("building split cache...", flush=True)

    # ---- gene indexing + row lists ----
    gidx = {}; rows = {s: [] for s in ["train", "val", "test"]}; exp_map = {}
    for s in ["train", "val", "test"]:
        for sp in gwd:
            if sp not in sets[s]:
                continue
            for g, gd in gwd[sp].items():
                key = (sp, g)
                if key not in gidx:
                    gidx[key] = len(gidx)
                gg = gidx[key]
                for c in gd["conditions"]:
                    e = c.get("expression")
                    if e is None:
                        continue
                    mv = c.get("media_vector")
                    mv = np.zeros(159, np.float32) if mv is None else np.asarray(mv, np.float32)
                    exk = (sp, c.get("media_name"))
                    if exk not in exp_map:
                        exp_map[exk] = len(exp_map)
                    rows[s].append((gg, float(e), mv, exp_map[exk]))
    NG = len(gidx); n_sp = len({sp for sp, _g in gidx})

    # ---- per-gene feature tables ----
    RAW = np.zeros((NG, 480), np.float32); CTX = np.zeros((NG, 480), np.float32)
    GIO = np.zeros((NG, 480), np.float32); GCLS = np.zeros((NG, 480), np.float32)
    DNA = np.zeros((NG, 65), np.float32)
    for (sp, g), gg in gidx.items():
        gi_info = gwd[sp][g]["gene_info"]
        # RAW (protein) and the gLM channels are indexed by their OWN gene order
        m = g2i.get(sp); pos = m.get(g) if m else None
        if pos is not None:
            RAW[gg] = RAWp[sp][pos]
        mG = g2iG.get(sp); posG = mG.get(g) if mG else None
        if posG is not None and sp in E2E:
            e = E2E[sp][posG]; GIO[gg] = e[:480]; CTX[gg] = e[480:]
        GCLS[gg] = CLSp.get(sp, np.zeros(480, np.float32))
        DNA[gg]  = dnasum(gi_info.get("gene_sequence_dna"))

    to = lambda a: torch.from_numpy(a).to(DEVICE)
    RAW, CTX, GIO, GCLS, DNA = to(RAW), to(CTX), to(GIO), to(GCLS), to(DNA)

    # ---- row tensors ----
    GI = {}; Y = {}; MED = {}; EXP = {}
    for s in ["train", "val", "test"]:
        r = rows[s]
        GI[s] = torch.tensor([x[0] for x in r], dtype=torch.long, device=DEVICE)
        Y[s]  = torch.tensor([x[1] for x in r], dtype=torch.float32, device=DEVICE)
        MED[s] = torch.from_numpy(np.stack([x[2] for x in r])).float().to(DEVICE) if r \
                 else torch.zeros(0, 159, device=DEVICE)
        EXP[s] = np.array([x[3] for x in r], dtype=np.int64)

    train_gids = torch.unique(GI["train"])
    if train_gids.numel() < 97:
        raise ValueError("Retrieval requires at least 97 labelled training genes (96 neighbors excluding self).")
    if any(Y[s].numel() < 2 for s in ("train", "val", "test")):
        raise ValueError("Each split requires at least two labelled rows; query rows use placeholder labels.")

    def gene_means():
        cnt = torch.zeros(NG, device=DEVICE); s = torch.zeros(NG, device=DEVICE)
        cnt.index_add_(0, GI["train"], torch.ones_like(Y["train"]))
        s.index_add_(0, GI["train"], Y["train"])
        return s / torch.clamp(cnt, min=1)
    train_mean = gene_means()

    # ---- per-(gene, medium) expression matrix (exact-medium lookup) ----
    all_med = torch.cat([MED[s] for s in ["train", "val", "test"]], 0)
    media_vectors, inv = torch.unique(all_med, dim=0, return_inverse=True); n_med = int(inv.max()) + 1
    off, MID = 0, {}
    for s in ["train", "val", "test"]:
        n = MED[s].shape[0]; MID[s] = inv[off:off + n]; off += n
    flat = GI["train"] * n_med + MID["train"]
    Msum = torch.zeros(NG, n_med, device=DEVICE); Mcnt = torch.zeros(NG, n_med, device=DEVICE)
    Msum.view(-1).index_add_(0, flat, Y["train"]); Mcnt.view(-1).index_add_(0, flat, torch.ones_like(Y["train"]))
    Mexpr = Msum / torch.clamp(Mcnt, min=1); Mhas = Mcnt > 0

    # Fit train-only feature statistics once; export the same values for queries.
    feature_stats = {}
    def standardized(name, T):
        if name not in feature_stats:
            sub = T[train_gids].float()
            feature_stats[name] = (sub.mean(0, keepdim=True), sub.std(0, keepdim=True) + 1e-6)
        mean, std = feature_stats[name]
        return (T.float() - mean) / std

    def zview(name, T):
        Tz = standardized(name, T)
        return (Tz / (Tz.shape[1] ** 0.5)).to(RETR_DTYPE)

    def knn(tables, name, k=96, chunk=8192, keytile=65536):
        X = torch.cat(tables, 1); Xn = (X / (X.norm(dim=1, keepdim=True) + 1e-8)).to(RETR_DTYPE)
        K = Xn[train_gids]; Ntr = train_gids.numel()
        if export is not None:
            print(f"  saving reference index: {name} ({Ntr:,} genes)", flush=True)
            np.save(export / f"keys_{name}.npy", K.detach().cpu().numpy())
        ids = torch.zeros(NG, k, dtype=torch.long, device=DEVICE); sims = torch.zeros(NG, k, device=DEVICE)
        q_all = torch.arange(NG, device=DEVICE)
        for cs in range(0, NG, chunk):
            if cs == 0 or cs // chunk % 20 == 0:
                print(f"  knn {name}: {cs:,}/{NG:,} genes", flush=True)
            q = Xn[cs:cs + chunk]; b = q.shape[0]; qid = q_all[cs:cs + chunk]
            top_v = torch.full((b, k + 1), -1e4, device=DEVICE, dtype=RETR_DTYPE)
            top_i = torch.zeros(b, k + 1, dtype=torch.long, device=DEVICE)
            for ks in range(0, Ntr, keytile):
                sim = q @ K[ks:ks + keytile].t(); kk = min(k + 1, sim.shape[1])
                v, i = torch.topk(sim, kk, 1); i = i + ks
                cv = torch.cat([top_v, v], 1); ci = torch.cat([top_i, i], 1)
                top_v, sel = torch.topk(cv, k + 1, 1); top_i = torch.gather(ci, 1, sel); del sim
            is_self = (train_gids[top_i] == qid.unsqueeze(1)); top_v = top_v.float().masked_fill(is_self, -1e4)
            v2, s2 = torch.topk(top_v, k, 1)
            ids[cs:cs + b] = train_gids[torch.gather(top_i, 1, s2)]; sims[cs:cs + b] = v2
        return ids, sims

    # standardized head tables (half) AND retrieval z-views are both computed from the
    # float32 raw tables; do them first so the float32 tables can be freed BEFORE the
    # memory-heavy kNN (keeps the GPU peak safe for the larger held-out training folds).
    RAWn, CTXn = standardized('raw', RAW).half(), standardized('ctx', CTX).half()
    GIOn, GCLSn = standardized('gio', GIO).half(), standardized('cls', GCLS).half()
    DNAn = standardized('dna', DNA).half()
    vRAW, vCTX, vGIO = zview('raw', RAW), zview('ctx', CTX), zview('gio', GIO)
    vGCLS, vDNA = zview('cls', GCLS), zview('dna', DNA)
    if export is not None:
        np.savez(export / 'feature_stats.npz', **{
            name + suffix: tensor.detach().cpu().numpy()
            for name, pair in feature_stats.items() for suffix, tensor in zip(('_mean', '_std'), pair)})
    del RAW, CTX, GIO, GCLS, DNA
    torch.cuda.empty_cache()

    print("  knn full / dual / multiview ...", flush=True)
    full_views = [vRAW, vCTX, vGIO, vGCLS, vDNA]
    NBR_F, SIM_F = knn(full_views, "full", k=96)                    # RAW+CTX+GIO+GCLS+DNA
    NBR_D, SIM_D = knn(full_views[:3], "dual", k=96)                # dual key (RAW+CTX+GIO)
    NBR_V0, SIM_V0 = knn([vRAW], "raw", k=96)                      # single-view [RAW]
    NBR_V2, SIM_V2 = knn([vGIO], "gio", k=96)                      # single-view [GIO]
    del vRAW, vCTX, vGIO, vGCLS, vDNA
    torch.cuda.empty_cache()

    if export is not None:
        for name, tensor in [('train_mean', train_mean[train_gids]), ('expression', Mexpr[train_gids]),
                             ('has_expression', Mhas[train_gids]), ('media_vectors', media_vectors)]:
            np.save(export / (name + '.npy'), tensor.detach().cpu().numpy())
        inv_gidx = {i: key for key, i in gidx.items()}
        with (export / 'reference_genes.tsv').open('w', encoding='utf-8') as handle:
            handle.write('index\tgenome_id\tgene_id\n')
            for i, gg in enumerate(train_gids.tolist()):
                handle.write(f'{i}\t{inv_gidx[gg][0]}\t{inv_gidx[gg][1]}\n')
        metadata = {'reference_genes': int(train_gids.numel()),
                    'reference_genome_ids': sorted(sets['train']),
                    'retrieval_dtype': 'float16' if RETR_DTYPE == torch.float16 else 'float32',
                    'feature_order': ['raw', 'ctx', 'gio', 'cls', 'dna']}
        (export / 'reference_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    print(f"  cache built  NG={NG:,}  n_sp={n_sp}  {time.time()-t0:.0f}s", flush=True)
    return dict(
        NG=NG, GI=GI, Y=Y, MED=MED, EXP=EXP, MID=MID, n_med=n_med, gidx=gidx,
        train_gids=train_gids, train_mean=train_mean, Mexpr=Mexpr, Mhas=Mhas,
        RAWn=RAWn, CTXn=CTXn, GIOn=GIOn, GCLSn=GCLSn, DNAn=DNAn,
        NBR_F=NBR_F, SIM_F=SIM_F, NBR_D=NBR_D, SIM_D=SIM_D,
        NBR_V0=NBR_V0, SIM_V0=SIM_V0, NBR_V2=NBR_V2, SIM_V2=SIM_V2,
    )


# ═══════════════════════════ derive_bundle ═══════════════════════════
def derive_bundle(cache):
    """Given a split cache, retrieval-decompose (with same-medium expression shift)
    and assemble the head feature table.  Medium conditioning is always on.
    Returns a bundle consumed by ``train_head``."""
    C = cache; GI = C["GI"]; Y = C["Y"]; train_mean = C["train_mean"]
    Mexpr = C["Mexpr"]; Mhas = C["Mhas"]; MID = C["MID"]

    NBR96, SIM96 = C["NBR_F"], C["SIM_F"]          # full enriched retrieval key

    def media_expr(nb, m):
        return Mexpr[nb, m.unsqueeze(1)], Mhas[nb, m.unsqueeze(1)]

    def retr_dec(nbr, sim, split, k, temp, bs=RETR_BS):
        gi = GI[split]; mid = MID[split]; N = gi.shape[0]; out = torch.zeros(N, device=DEVICE)
        for b in range(0, N, bs):
            g = gi[b:b + bs]; e = b + g.shape[0]; m = mid[b:b + bs]
            nb = nbr[g][:, :k]; sims = sim[g][:, :k]; nmean = train_mean[nb]
            rp = (torch.softmax(sims / temp, 1) * nmean).sum(1)
            expr, has = media_expr(nb, m)
            w = torch.softmax(sims / temp + torch.where(has, 0., -1e4), 1)
            wsum = torch.clamp((w * has).sum(1), min=1e-6); c = has.any(1).float()
            out[b:e] = rp + (w * (expr - nmean) * has).sum(1) / wsum * c
        return out

    def retr_full_dec(split, k, temp, bs=RETR_BS):
        gi = GI[split]; mid = MID[split]; N = gi.shape[0]
        cr = torch.zeros(N, device=DEVICE); cd = torch.zeros(N, device=DEVICE); cov = torch.zeros(N, device=DEVICE)
        for b in range(0, N, bs):
            g = gi[b:b + bs]; e = b + g.shape[0]; m = mid[b:b + bs]
            nb = NBR96[g][:, :k]; sims = SIM96[g][:, :k]; nmean = train_mean[nb]
            rp = (torch.softmax(sims / temp, 1) * nmean).sum(1)
            expr, has = media_expr(nb, m)
            w = torch.softmax(sims / temp + torch.where(has, 0., -1e4), 1)
            wsum = torch.clamp((w * has).sum(1), min=1e-6); c = has.any(1).float()
            cr[b:e] = ((w * expr * has).sum(1) / wsum) * c
            cd[b:e] = rp + (w * (expr - nmean) * has).sum(1) / wsum * c
            cov[b:e] = c
        return cr, cd, cov

    # grid-select (k, temperature) on val
    best = (-1, None)
    for k in [16, 32, 48, 96]:
        for temp in [0.02, 0.03, 0.05, 0.1]:
            vr = pear(retr_dec(NBR96, SIM96, "val", k, temp), Y["val"])
            if vr > best[0]:
                best = (vr, (k, temp))
    K, T = best[1]
    RETR = {s: retr_full_dec(s, K, T) for s in ["train", "val", "test"]}

    # multi-(k, temperature) prior ensemble
    PRIOR = {s: torch.stack([retr_dec(NBR96, SIM96, s, k, t) for (k, t) in MULTI_KT]).mean(0)
             for s in ["train", "val", "test"]}

    # head feature tables
    retr_full = (torch.softmax(SIM96[:, :K] / T, 1) * train_mean[NBR96[:, :K]]).sum(1)
    top_sim = SIM96[:, 0].float()

    cols = [C["RAWn"], C["CTXn"], C["GIOn"], C["GCLSn"], C["DNAn"]]
    retr_mu = retr_full[C['train_gids']].mean()
    retr_sd = retr_full[C['train_gids']].std() + 1e-6
    cols.append(((retr_full - retr_mu) / retr_sd).unsqueeze(1).half())
    GTAB = torch.cat(cols, 1)
    # sparse channels can z-score to values that overflow fp16 -> inf on some folds;
    # sanitize so a degenerate feature entry can't poison the whole head.
    GTAB = torch.nan_to_num(GTAB, nan=0.0, posinf=0.0, neginf=0.0)

    # dual + single-view retrieval decompositions (extra ridge-stack columns)
    retr_dual = {s: retr_dec(C["NBR_D"], C["SIM_D"], s, K, T) for s in ["train", "val", "test"]}
    retr_mv = []
    for nbr, sim in [(C["NBR_V0"], C["SIM_V0"]), (C["NBR_V2"], C["SIM_V2"])]:
        retr_mv.append({s: retr_dec(nbr, sim, s, K, T) for s in ["train", "val", "test"]})

    # medium head input (log1p + train-z)
    MED = C["MED"]
    rlog = {s: torch.log1p(torch.clamp(MED[s], min=0)) for s in MED}
    mu, sd = rlog["train"].mean(0, keepdim=True), rlog["train"].std(0, keepdim=True) + 1e-6
    MEDn = {s: (rlog[s] - mu) / sd for s in rlog}

    cr_mu, cr_sd = RETR["train"][0].mean(), RETR["train"][0].std() + 1e-6
    cd_mu, cd_sd = RETR["train"][1].mean(), RETR["train"][1].std() + 1e-6
    RETR3 = {s: torch.stack([(RETR[s][0] - cr_mu) / cr_sd, (RETR[s][1] - cd_mu) / cd_sd, RETR[s][2]], 1)
             for s in ["train", "val", "test"]}

    return dict(GI=GI, Y=Y, EXP=C["EXP"], NG=C["NG"], train_gids=C["train_gids"],
                train_mean=train_mean, GTAB=GTAB, MEDn=MEDn, RETR=RETR, RETR3=RETR3, PRIOR=PRIOR,
                retr_full=retr_full, top_sim=top_sim, retr_dual=retr_dual, retr_mv=retr_mv,
                kT=(K, T), transforms=dict(retr_mean=retr_mu, retr_std=retr_sd,
                    media_mean=mu, media_std=sd, cr_mean=cr_mu, cr_std=cr_sd,
                    cd_mean=cd_mu, cd_std=cd_sd))


# ═══════════════════════════ head + stacking ═══════════════════════════
class _ResBlock(nn.Module):
    def __init__(s, dd, p=0.1):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(dd), nn.Linear(dd, dd), nn.GELU(),
                              nn.Dropout(p), nn.Linear(dd, dd))
    def forward(s, x):
        return x + s.net(x)


class _Head(nn.Module):
    def __init__(s, gd, md, hid=768, nb=4, p=0.1, dm=128):
        super().__init__()
        s.gene = nn.Sequential(nn.Linear(gd, hid), nn.GELU())
        s.blocks = nn.ModuleList([_ResBlock(hid, p) for _ in range(nb)])
        s.media = nn.Sequential(nn.Linear(md, dm), nn.LayerNorm(dm), nn.GELU(),
                                nn.Dropout(p), nn.Linear(dm, dm), nn.GELU())
        s.film = nn.Linear(dm, 2 * hid)
        s.retr = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Dropout(p), nn.Linear(64, hid))
        s.out = nn.Sequential(nn.LayerNorm(hid + dm), nn.Linear(hid + dm, hid), nn.GELU(),
                              nn.Dropout(p), nn.Linear(hid, 1))
    def forward(s, gf, mv, retr3, prior):
        h = s.gene(gf.float())
        for blk in s.blocks:
            h = blk(h)
        h = h + s.retr(retr3); m = s.media(mv); ga, be = s.film(m).chunk(2, -1)
        h = h * (1 + torch.tanh(ga)) + be
        return s.out(torch.cat([h, m], -1)).squeeze(-1)


def train_head(bundle, hp=None, seeds=8, artifact_dir=None):
    """8-seed FiLM residual head (predicts y - prior) + LightGBM + ridge stack.

    Returns a dict with:
        pred_test           final stacked prediction, one value per test row (np.array)
        y_test              measured expression per test row (np.array)
        gene_idx_test       gene index per test row (np.array; maps rows -> genes,
                            invertible via cache["gidx"])
        test_r              pooled Pearson r over all test rows
        within_condition_r  mean Pearson r within each (species, medium) condition
    """
    export = Path(artifact_dir) if artifact_dir is not None else None
    if export is not None:
        (export / 'heads').mkdir(parents=True, exist_ok=True)
    HP = dict(HP_DEFAULT); HP.update(hp or {})
    # explicit hp["seeds"] takes precedence (preserves tuned config); else the arg
    n_seeds = HP["seeds"] if (hp and "seeds" in hp) else seeds
    HP["seeds"] = n_seeds
    if n_seeds < 1 or HP["epochs"] < 1 or HP["bs"] < 1:
        raise ValueError("seeds, epochs and bs must be positive")
    B = bundle; GI = B["GI"]; Y = B["Y"]; GTAB = B["GTAB"]; MEDn = B["MEDn"]
    RETR3 = B["RETR3"]; PRIOR = B["PRIOR"]; RETR = B["RETR"]; retr_full = B["retr_full"]
    top_sim = B["top_sim"]; train_mean = B["train_mean"]; EXP = B["EXP"]

    def _train_seed(seed):
        print(f"[head] seed {seed + 1}/{n_seeds}: training", flush=True)
        torch.manual_seed(seed)
        model = _Head(GTAB.shape[1], MEDn["train"].shape[1],
                      hid=HP["hid"], nb=HP["nb"], p=HP["dropout"], dm=HP["dm"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])

        def predict(sp, chunk=131072):
            model.eval(); out = torch.empty(Y[sp].shape[0], device=DEVICE)
            with torch.no_grad():
                for b in range(0, out.shape[0], chunk):
                    gi = GI[sp][b:b + chunk]
                    out[b:b + chunk] = model(GTAB[gi], MEDn[sp][b:b + chunk],
                                             RETR3[sp][b:b + chunk], PRIOR[sp][b:b + chunk]) \
                        + PRIOR[sp][b:b + chunk]
            return out

        N = Y["train"].shape[0]; best = -1; bst = None; stale = 0; bs = HP["bs"]
        for ep in range(1, HP["epochs"] + 1):
            model.train(); perm = torch.randperm(N, device=DEVICE)
            for b in range(0, N, bs):
                ix = perm[b:b + bs]; gi = GI["train"][ix]; opt.zero_grad(set_to_none=True)
                tgt = Y["train"][ix] - PRIOR["train"][ix]
                F.smooth_l1_loss(model(GTAB[gi], MEDn["train"][ix], RETR3["train"][ix],
                                       PRIOR["train"][ix]), tgt).backward()
                opt.step()
            sched.step(); vr = pear(predict("val"), Y["val"])
            if ep == 1 or ep % 5 == 0:
                print(f"[head] seed={seed} epoch={ep}/{HP['epochs']} val_r={vr:.5f}", flush=True)
            if vr > best:
                best, stale = vr, 0
                bst = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= HP["patience"]:
                    break
        if bst is None:
            raise RuntimeError("No finite validation checkpoint; check labels, features and hyperparameters")
        model.load_state_dict(bst)
        pv, pt = predict("val"), predict("test")
        if export is not None and torch.isfinite(pv).all() and torch.isfinite(pt).all():
            torch.save({k: v.detach().cpu() for k, v in bst.items()}, export / 'heads' / f'seed_{seed}.pt')
        print(f"[head] seed={seed} complete; best_val_r={best:.5f}", flush=True)
        return pv, pt

    pvs, pts, accepted_seeds = [], [], []
    for sd in range(n_seeds):
        pv, pt = _train_seed(sd)
        if torch.isfinite(pv).all() and torch.isfinite(pt).all():   # drop diverged seeds
            pvs.append(pv); pts.append(pt); accepted_seeds.append(sd)
    if not pvs:                                                      # all seeds diverged (rare)
        pv, pt = _train_seed(n_seeds)
        if not torch.isfinite(pv).all() or not torch.isfinite(pt).all():
            raise RuntimeError('All neural head seeds diverged')
        pvs.append(pv); pts.append(pt); accepted_seeds.append(n_seeds)
    hv = torch.stack(pvs).mean(0); ht = torch.stack(pts).mean(0)

    print("[stack] fitting LightGBM", flush=True)
    # LightGBM stack member
    gbm_v = gbm_t = None
    try:
        import lightgbm as lgb
        def tab(sp):
            return torch.cat([RETR3[sp], PRIOR[sp].unsqueeze(1),
                              retr_full[GI[sp]].unsqueeze(1), MEDn[sp]], 1).cpu().numpy()
        gbm = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.6, verbosity=-1,
                                **({"n_jobs": int(os.environ["ACTEXPR_THREADS"])}
                                   if "ACTEXPR_THREADS" in os.environ else {}))
        gbm.fit(tab("train"), Y["train"].cpu().numpy(),
                eval_set=[(tab("val"), Y["val"].cpu().numpy())],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        gbm_v = torch.tensor(gbm.predict(tab("val")), device=DEVICE, dtype=torch.float32)
        gbm_t = torch.tensor(gbm.predict(tab("test")), device=DEVICE, dtype=torch.float32)
    except Exception as ex:
        raise RuntimeError("LightGBM failed; the canonical model requires this stack member") from ex

    if export is not None:
        gbm.booster_.save_model(str(export / 'lightgbm.txt'))
    print("[stack] fitting ridge on validation rows", flush=True)
    # ridge stacking (fit on val)
    def base_mat(split, head_pred, gbm_pred):
        cols = [head_pred, PRIOR[split], RETR[split][0], retr_full[GI[split]], train_mean[GI[split]]]
        cols.append(B["retr_dual"][split])
        for cfm in B["retr_mv"]:
            cols.append(cfm[split])
        cols.append(top_sim[GI[split]]); cols.append(RETR[split][2])
        if gbm_pred is not None:
            cols.append(gbm_pred)
        return torch.stack(cols, 1)
    Bv = base_mat("val", hv, gbm_v); Bt = base_mat("test", ht, gbm_t)
    mu_, sd_ = Bv.mean(0, keepdim=True), Bv.std(0, keepdim=True) + 1e-6
    Bv1 = torch.cat([(Bv - mu_) / sd_, torch.ones(Bv.shape[0], 1, device=DEVICE)], 1)
    Bt1 = torch.cat([(Bt - mu_) / sd_, torch.ones(Bt.shape[0], 1, device=DEVICE)], 1)
    lam = HP["ridge_lam_scale"] * Bv1.shape[1]
    wS = torch.linalg.solve(Bv1.t() @ Bv1 + lam * torch.eye(Bv1.shape[1], device=DEVICE), Bv1.t() @ Y["val"])
    stack_test = Bt1 @ wS

    if export is not None:
        np.savez(export / 'stack.npz', mean=mu_.detach().cpu().numpy(),
                 std=sd_.detach().cpu().numpy(), weights=wS.detach().cpu().numpy())
        np.savez(export / 'transforms.npz', **{name: value.detach().cpu().numpy()
                                             for name, value in B['transforms'].items()})
        metadata = {'hp': HP, 'seeds': accepted_seeds, 'gene_dim': GTAB.shape[1],
                    'media_dim': MEDn['train'].shape[1], 'kT': list(B['kT']),
                    'prior_grid': [list(pair) for pair in MULTI_KT]}
        (export / 'fit_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    pred_test = stack_test.detach().cpu().numpy()
    y_test = Y["test"].detach().cpu().numpy()
    return {
        "pred_test": pred_test,
        "y_test": y_test,
        "gene_idx_test": GI["test"].detach().cpu().numpy(),
        "test_r": pear(stack_test, Y["test"]),
        "within_condition_r": within_cond(pred_test, y_test, EXP["test"]),
    }
