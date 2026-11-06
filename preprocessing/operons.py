"""UniOP-style distance posterior ported from Tools/Run_UniOP.ipynb.

This preserves the project's local KDE recipe (seed 0, bandwidth .03,
posterior threshold .5); it is not a dependency on an external UniOP package.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def fit_kde_u(u, bandwidth=0.03):
    u = np.clip(np.asarray(u, dtype=float), 1e-6, 1 - 1e-6)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(u.reshape(-1, 1))
    def density(uq):
        uq = np.clip(np.asarray(uq, dtype=float), 1e-6, 1 - 1e-6)
        return np.exp(kde.score_samples(uq.reshape(-1, 1)))
    return density

def adjacent_pairs(genes):
    rows = []
    for i in range(len(genes) - 1):
        g1, g2 = genes[i], genes[i+1]
        d  = g2["start"] - g1["end"]
        s1, s2 = g1["strand"], g2["strand"]
        if s1 == s2:
            ptype = "same"
        elif s1 == -1 and s2 == +1:
            ptype = "divergent"
        elif s1 == +1 and s2 == -1:
            ptype = "convergent"
        else:
            ptype = "opposite"
        rows.append({
            "contig":    g1["contig"],
            "g1": g1["gene_id"], "g2": g2["gene_id"],
            "g1_start":  g1["start"], "g1_end": g1["end"], "g1_strand": s1,
            "g2_start":  g2["start"], "g2_end": g2["end"], "g2_strand": s2,
            "d":         int(d),
            "ptype":     ptype,
        })
    return pd.DataFrame(rows)

def ecdf_transform(x, ref_sorted):
    n = ref_sorted.size
    r = np.searchsorted(ref_sorted, x, side="right")
    return (r - 0.5) / n

def uniop_posterior_for_genome(df_pairs, n_sample=10_000, kde_bw=0.03, seed=0):
    rng = np.random.default_rng(seed)

    M = len(df_pairs)
    O = int((df_pairs["ptype"] != "same").sum())
    denom = M - O
    q = (M - 2*O) / denom if denom > 0 else 0.5
    q = float(np.clip(q, 1e-6, 1 - 1e-6))

    d_same = df_pairs.loc[df_pairs["ptype"] == "same",      "d"].to_numpy()
    d_div  = df_pairs.loc[df_pairs["ptype"] == "divergent", "d"].to_numpy()
    d_conv = df_pairs.loc[df_pairs["ptype"] == "convergent","d"].to_numpy()

    if len(d_div) < 10 or len(d_conv) < 10:
        d_nonop = df_pairs.loc[df_pairs["ptype"] != "same", "d"].to_numpy()
        if len(d_nonop) < 50:
            raise ValueError("Not enough opposite-strand pairs to estimate non-operonic distribution.")
    else:
        div_s  = rng.choice(d_div,  size=n_sample, replace=True)
        conv_s = rng.choice(d_conv, size=n_sample, replace=True)
        d_nonop = 0.5 * (div_s + conv_s)

    ref        = np.concatenate([d_same, d_nonop]).astype(float)
    ref_sorted = np.sort(ref)

    u_same  = ecdf_transform(d_same.astype(float),   ref_sorted)
    u_nonop = ecdf_transform(d_nonop.astype(float),  ref_sorted)

    p_same_u  = fit_kde_u(u_same,  bandwidth=kde_bw)
    p_nonop_u = fit_kde_u(u_nonop, bandwidth=kde_bw)

    eps    = 1e-12
    p_d    = np.maximum(p_same_u(u_same),  eps)
    p_d_z0 = np.maximum(p_nonop_u(u_same), eps)

    posterior = np.clip(1.0 - (p_d_z0 / p_d) * (1.0 - q), 0.0, 1.0)

    df_same = df_pairs[df_pairs["ptype"] == "same"].copy()
    df_same["posterior_operon"] = posterior

    def robust_cutoff(t):
        dd = df_same.loc[df_same["posterior_operon"] >= t, "d"]
        return float(np.percentile(dd.to_numpy(), 95)) if len(dd) > 0 else np.nan

    summary = {
        "M_pairs":            M,
        "O_opposite_pairs":   O,
        "q_prior":            q,
        "n_same":             int(len(d_same)),
        "n_div":              int(len(d_div)),
        "n_conv":             int(len(d_conv)),
        "cutoff_d_p95_t0.5":  robust_cutoff(0.5),
        "cutoff_d_p95_t0.8":  robust_cutoff(0.8),
        "cutoff_d_p95_t0.9":  robust_cutoff(0.9),
    }
    return df_same, summary

def build_operons_from_same_edges(genes, df_same_edges, t=0.5):
    keep = {(r.g1, r.g2): (r.posterior_operon >= t)
            for r in df_same_edges.itertuples(index=False)}

    operons = []
    current        = [genes[0]["gene_id"]]
    current_start  = genes[0]["start"]
    current_end    = genes[0]["end"]
    current_strand = genes[0]["strand"]

    for i in range(len(genes) - 1):
        g1 = genes[i]["gene_id"]
        g2 = genes[i+1]["gene_id"]

        if keep.get((g1, g2), False):
            current.append(g2)
            current_end = max(current_end, genes[i+1]["end"])
        else:
            operons.append({
                "contig":  genes[i]["contig"],
                "strand":  current_strand,
                "start":   current_start,
                "end":     current_end,
                "n_genes": len(current),
                "genes":   current,
            })
            current        = [g2]
            current_start  = genes[i+1]["start"]
            current_end    = genes[i+1]["end"]
            current_strand = genes[i+1]["strand"]

    operons.append({
        "contig":  genes[-1]["contig"],
        "strand":  current_strand,
        "start":   current_start,
        "end":     current_end,
        "n_genes": len(current),
        "genes":   current,
    })
    return operons

def infer_operons(genome, mode='uniop'):
    contigs = {}
    for gene in genome['genes']:
        contigs.setdefault(gene['contig_id'], []).append({**gene, 'contig': gene['contig_id']})
    tables = [adjacent_pairs(genes) for genes in contigs.values() if len(genes) >= 2]
    if mode == 'uniop':
        if not tables:
            raise ValueError('UniOP needs adjacent CDS pairs on at least one contig')
        pairs = pd.concat(tables, ignore_index=True)
        if not (pairs['ptype'] == 'same').any():
            raise ValueError('UniOP requires same-strand pairs; use --operon-mode singletons explicitly for a diagnostic fragment')
        try:
            edges, summary = uniop_posterior_for_genome(pairs)
        except ValueError as exc:
            raise ValueError(str(exc) + ' Supply a complete genome, or explicitly use --operon-mode singletons for a diagnostic fragment.') from exc
    elif mode == 'singletons':
        edges = pd.DataFrame(columns=['contig','g1','g2','posterior_operon'])
        summary = {'mode': 'singletons', 'note': 'Explicit diagnostic mode; not the trained UniOP recipe'}
    else:
        raise ValueError('operon mode must be uniop or singletons')
    ops = []
    for contig, genes in contigs.items():
        subset = edges[edges['contig'] == contig]
        for i, op in enumerate(build_operons_from_same_edges(genes, subset), 1):
            ops.append({**op, 'genome': genome['genome_id'],
                        'operon_id': f"{genome['genome_id']}|{contig}|op{i:05d}"})
    # JSON does not have a portable NaN representation.
    summary = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k,v in summary.items()}
    summary['mode'] = mode
    return ops, summary
