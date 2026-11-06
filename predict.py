"""Inference only: load a fitted model and predict expression for a new genome."""
import numpy as np


def predict(genome, medium, new_embeddings, model_dir=None, device=None):
    """Return {gene_id: log_expression}; never read training data or fit models.

    For multiple genomes/media, construct serving.ExpressionPredictor once and
    reuse its predict() method to avoid reloading head weights.
    """
    import config
    from model_artifacts import check_model
    directory = check_model(model_dir or config.MODEL_DIR)
    from serving import ExpressionPredictor
    return ExpressionPredictor(directory, device=device).predict(genome, medium, new_embeddings)


def _validate_query(genome, medium, embeddings, reference_ids=()):
    if genome.genome_id in reference_ids:
        raise ValueError('Query genome_id already occurs in reference; use a new genome identifier')
    gids = [g.gene_id for g in genome.genes]
    if len(gids) < 1 or len(gids) != len(set(gids)):
        raise ValueError('Query must have at least one gene with unique gene_id values')
    if medium.vector is None:
        raise ValueError('Medium.vector is required; use the reference media vocabulary and order')
    vector = np.asarray(medium.vector, dtype=np.float32)
    if vector.shape != (159,) or not np.isfinite(vector).all():
        raise ValueError('Medium.vector must contain 159 finite values in reference component order')
    sid = genome.genome_id
    for channel, width, order in [('esm_emb', 480, 'esm_order'), ('glm_emb', 960, 'glm_order')]:
        if sid not in embeddings.get(channel, {}) or sid not in embeddings.get(order, {}):
            raise ValueError(f'Missing {channel}/{order} for query {sid}')
        ids = list(embeddings[order][sid])
        arr = np.asarray(embeddings[channel][sid])
        if len(ids) != len(set(ids)) or not set(gids).issubset(ids):
            raise ValueError(f'{order} must contain all query genes exactly once')
        if arr.shape != (len(ids), width) or not np.isfinite(arr).all():
            raise ValueError(f'{channel} must have shape (len({order}), {width}) with finite values')
    cls = np.asarray(embeddings.get('glm_cls', {}).get(sid, []))
    if cls.shape != (480,) or not np.isfinite(cls).all():
        raise ValueError('glm_cls must contain a finite 480-dimensional query vector')
