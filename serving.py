"""Frozen-model inference with query-only, tiled nearest-neighbor search."""
import json
import os
import time

import numpy as np
import torch

from expression_model import _Head, dnasum
from model_artifacts import check_model
from predict import _validate_query

VIEWS = {'full': ('raw', 'ctx', 'gio', 'cls', 'dna'),
         'dual': ('raw', 'ctx', 'gio'), 'raw': ('raw',), 'gio': ('gio',)}


class ExpressionPredictor:
    """Load once, reuse for any number of new genomes or media. No fitting.

    Reference keys remain memory-mapped on CPU; only one key tile and the query
    features are placed on the GPU. query_chunk/key_tile bound the similarity
    matrix size independently of the total reference corpus.
    """
    def __init__(self, model_dir, device=None, query_chunk=256, key_tile=65536, verbose=True):
        directory = check_model(model_dir)
        self.device = torch.device(device or os.environ.get('ACTEXPR_DEVICE') or
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; use --device cpu')
        if query_chunk < 1 or key_tile < 1:
            raise ValueError('query_chunk and key_tile must be positive')
        self.query_chunk, self.key_tile, self.verbose = query_chunk, key_tile, verbose
        self.meta = json.loads((directory / 'reference_metadata.json').read_text())
        self.fit = json.loads((directory / 'fit_metadata.json').read_text())
        self.storage_dtype = getattr(torch, self.meta['retrieval_dtype'])
        # CPU half-precision matrix multiplication is unsupported in torch 2.0.
        self.matmul_dtype = self.storage_dtype if self.device.type == 'cuda' else torch.float32
        self.keys = {name: np.load(directory / f'keys_{name}.npy', mmap_mode='r', allow_pickle=False)
                     for name in VIEWS}
        self.means = np.load(directory / 'train_mean.npy', mmap_mode='r', allow_pickle=False)
        self.expression = np.load(directory / 'expression.npy', mmap_mode='r', allow_pickle=False)
        self.has_expression = np.load(directory / 'has_expression.npy', mmap_mode='r', allow_pickle=False)
        self.media = np.load(directory / 'media_vectors.npy', allow_pickle=False)
        n = self.meta['reference_genes']
        if n < 97 or self.means.shape != (n,) or self.expression.shape != (n, len(self.media)):
            raise ValueError('Invalid reference expression shapes')
        if self.has_expression.shape != self.expression.shape or self.media.shape[1:] != (159,):
            raise ValueError('Invalid reference medium shapes')
        for name, dim in [('full', 1985), ('dual', 1440), ('raw', 480), ('gio', 480)]:
            if self.keys[name].shape != (n, dim):
                raise ValueError(f'Invalid reference index: {name}')
        self.stats = self._npz(directory / 'feature_stats.npz')
        self.transforms = self._npz(directory / 'transforms.npz')
        self.stack = self._npz(directory / 'stack.npz')
        hp = self.fit['hp']
        self.heads = []
        for seed in self.fit['seeds']:
            head = _Head(self.fit['gene_dim'], self.fit['media_dim'], hid=hp['hid'],
                         nb=hp['nb'], p=hp['dropout'], dm=hp['dm'])
            head.load_state_dict(torch.load(directory / 'heads' / f'seed_{seed}.pt',
                                            map_location='cpu', weights_only=True))
            self.heads.append(head.to(self.device).eval())
        import lightgbm as lgb
        self.gbm = lgb.Booster(model_file=str(directory / 'lightgbm.txt'))
        self._log(f'[model] loaded {len(self.heads)} heads; {n:,} reference genes; device={self.device}')

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _tensor(self, array):
        return torch.tensor(np.asarray(array), device=self.device)

    def _npz(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            return {name: self._tensor(arrays[name]) for name in arrays.files}

    def _knn(self, query, keys):
        """Top-96, using the same top-97 candidate merges as training."""
        n = len(query)
        query = query.to(self.matmul_dtype)
        top_v = torch.full((n, 97), -1e4, dtype=self.matmul_dtype, device=self.device)
        top_i = torch.zeros((n, 97), dtype=torch.long, device=self.device)
        for start in range(0, len(keys), self.key_tile):
            # Copy only this tile; the complete index is never copied to the GPU.
            tile = self._tensor(keys[start:start + self.key_tile]).to(self.matmul_dtype)
            for b in range(0, n, self.query_chunk):
                end = min(b + self.query_chunk, n)
                sim = query[b:end] @ tile.t()
                v, i = torch.topk(sim, min(97, sim.shape[1]), 1)
                values = torch.cat([top_v[b:end], v], 1)
                indices = torch.cat([top_i[b:end], i + start], 1)
                top_v[b:end], selected = torch.topk(values, 97, 1)
                top_i[b:end] = torch.gather(indices, 1, selected)
                del sim
            del tile
        sims, selected = torch.topk(top_v.float(), 96, 1)
        return torch.gather(top_i, 1, selected).cpu().numpy(), sims

    def _retrieve(self, neighbors, sims, medium_index, k, temperature):
        nb = neighbors[:, :k]
        similarities = sims[:, :k]
        means = self._tensor(self.means[nb])
        rp = (torch.softmax(similarities / temperature, 1) * means).sum(1)
        if medium_index is None:
            zero = torch.zeros_like(rp)
            return rp, zero, rp, zero
        expr = self._tensor(self.expression[nb, medium_index])
        has = self._tensor(self.has_expression[nb, medium_index])
        weights = torch.softmax(similarities / temperature + torch.where(has, 0., -1e4), 1)
        wsum = (weights * has).sum(1).clamp(min=1e-6)
        coverage = has.any(1).float()
        cr = (weights * expr * has).sum(1) / wsum * coverage
        cd = rp + (weights * (expr - means) * has).sum(1) / wsum * coverage
        return rp, cr, cd, coverage

    @torch.inference_mode()
    def predict(self, genome, medium, new_embeddings):
        start = time.monotonic()
        _validate_query(genome, medium, new_embeddings, self.meta['reference_genome_ids'])
        sid = genome.genome_id
        def aligned(channel, order):
            positions = {gene: i for i, gene in enumerate(new_embeddings[order][sid])}
            values = np.asarray(new_embeddings[channel][sid], dtype=np.float32)
            return self._tensor(values[[positions[g.gene_id] for g in genome.genes]])
        raw = aligned('esm_emb', 'esm_order')
        glm = aligned('glm_emb', 'glm_order')
        cls = self._tensor(np.asarray(new_embeddings['glm_cls'][sid], dtype=np.float32))
        dna = self._tensor(np.stack([dnasum(g.dna_sequence) for g in genome.genes]))
        features = {'raw': raw, 'ctx': glm[:, 480:], 'gio': glm[:, :480],
                    'cls': cls.expand(len(genome.genes), -1), 'dna': dna}
        standardized, zviews = {}, {}
        for name, value in features.items():
            standardized[name] = (value - self.stats[name + '_mean']) / self.stats[name + '_std']
            zviews[name] = (standardized[name] / value.shape[1] ** 0.5).to(self.storage_dtype)
        vector = np.asarray(medium.vector, dtype=np.float32)
        matches = np.flatnonzero(np.all(self.media == vector, axis=1))
        mid = int(matches[0]) if len(matches) else None
        k, temperature = self.fit['kT']
        retrieval = {}
        for name, channels in VIEWS.items():
            self._log(f'[retrieval] {name}: {len(genome.genes):,} query genes')
            query = torch.cat([zviews[ch] for ch in channels], 1)
            query = (query / (query.norm(dim=1, keepdim=True) + 1e-8)).to(self.storage_dtype)
            nb, sims = self._knn(query, self.keys[name])
            retrieval[name] = self._retrieve(nb, sims, mid, k, temperature)
            if name == 'full':
                top_sim = sims[:, 0]
                prior = torch.stack([self._retrieve(nb, sims, mid, pk, pt)[2]
                                     for pk, pt in self.fit['prior_grid']]).mean(0)
        retr_full, cr, cd, coverage = retrieval['full']
        tr = self.transforms
        columns = [standardized[ch].half() for ch in VIEWS['full']]
        columns.append(((retr_full - tr['retr_mean']) / tr['retr_std']).unsqueeze(1).half())
        gtab = torch.nan_to_num(torch.cat(columns, 1), nan=0., posinf=0., neginf=0.)
        med = (torch.log1p(self._tensor(vector).clamp(min=0)) - tr['media_mean']) / tr['media_std']
        med = med.expand(len(genome.genes), -1)
        retr3 = torch.stack([(cr - tr['cr_mean']) / tr['cr_std'],
                             (cd - tr['cd_mean']) / tr['cd_std'], coverage], 1)
        self._log('[prediction] neural heads / LightGBM / ridge')
        head_outputs = []
        for head in self.heads:
            chunks = [head(gtab[b:b+self.query_chunk], med[b:b+self.query_chunk],
                           retr3[b:b+self.query_chunk], prior[b:b+self.query_chunk]) + prior[b:b+self.query_chunk]
                      for b in range(0, len(genome.genes), self.query_chunk)]
            head_outputs.append(torch.cat(chunks))
        head_mean = torch.stack(head_outputs).mean(0)
        table = torch.cat([retr3, prior.unsqueeze(1), retr_full.unsqueeze(1), med], 1).cpu().numpy()
        gbm = self._tensor(self.gbm.predict(table, num_threads=int(os.environ.get('ACTEXPR_THREADS', '4')))).float()
        base = torch.stack([head_mean, prior, cr, retr_full, torch.zeros_like(prior),
                            retrieval['dual'][2], retrieval['raw'][2], retrieval['gio'][2],
                            top_sim, coverage, gbm], 1)
        z = (base - self.stack['mean']) / self.stack['std']
        z = torch.cat([z, torch.ones(len(genome.genes), 1, device=self.device)], 1)
        predictions = (z @ self.stack['weights']).cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError('Non-finite prediction; check query features and fitted model')
        self._log(f'[done] {len(predictions):,} genes in {time.monotonic() - start:.1f}s')
        return {g.gene_id: float(p) for g, p in zip(genome.genes, predictions)}
