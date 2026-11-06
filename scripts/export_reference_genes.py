"""Write reference_genes.tsv (index row -> genome_id, gene_id) for a fitted model.

Rebuilds the retrieval-row order of expression_model.build_cache from the same
reference inputs, then checks it against the saved keys_raw.npy before writing.
Use this for model directories exported before reference_genes.tsv was saved.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def reference_order(unified_data, train_species):
    """Same gene indexing as build_cache: train genomes first, labelled genes only."""
    gwd = unified_data['gene_wise_data']
    train_species = set(train_species)
    keys = []
    for sp in gwd:
        if sp not in train_species:
            continue
        for g, gd in gwd[sp].items():
            if any(c.get('expression') is not None for c in gd['conditions']):
                keys.append((sp, g))
    return keys


def check_against_index(keys, model_dir, ref, n_check=2000, seed=0):
    stats = np.load(model_dir / 'feature_stats.npz')
    index = np.load(model_dir / 'keys_raw.npy', mmap_mode='r')
    if index.shape[0] != len(keys):
        raise ValueError(f'row count {len(keys):,} != keys_raw {index.shape[0]:,}')
    pos = {sp: {g: i for i, g in enumerate(ids)} for sp, ids in ref['esm_order'].items()}
    rows = np.random.default_rng(seed).choice(len(keys), min(n_check, len(keys)), replace=False)
    worst = 1.0
    for r in rows:
        sp, g = keys[r]
        p = pos.get(sp, {}).get(g)
        raw = ref['esm_emb'][sp][p] if p is not None else np.zeros(480, np.float32)
        z = (np.asarray(raw, np.float32) - stats['raw_mean'][0]) / stats['raw_std'][0]
        z = z / (np.linalg.norm(z) + 1e-8)
        k = np.asarray(index[r], np.float32)
        worst = min(worst, float(z @ k / (np.linalg.norm(k) + 1e-8)))
    if worst < 0.999:
        raise ValueError(f'reconstructed order does not match keys_raw (min cosine {worst:.4f})')
    return len(rows), worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default=config.MODEL_DIR)
    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / 'reference_metadata.json').read_text(encoding='utf-8'))
    import pickle
    def load(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    ref = {'unified_data': load(config.UNIFIED_DATA),
           'esm_emb': load(config.ESM_EMB), 'esm_order': load(config.ESM_ORDER)}
    keys = reference_order(ref['unified_data'], meta['reference_genome_ids'])
    if len(keys) != meta['reference_genes']:
        raise ValueError(f"{len(keys):,} genes != reference_metadata {meta['reference_genes']:,}")
    n, worst = check_against_index(keys, model_dir, ref)
    target = model_dir / 'reference_genes.tsv'
    with target.open('w', encoding='utf-8') as handle:
        handle.write('index\tgenome_id\tgene_id\n')
        for i, (sp, g) in enumerate(keys):
            handle.write(f'{i}\t{sp}\t{g}\n')
    print(f'{target}: {len(keys):,} rows; {n:,} sampled rows match keys_raw (min cosine {worst:.5f})')


if __name__ == '__main__':
    main()
