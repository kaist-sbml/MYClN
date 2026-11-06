"""Encode a named medium or a composition using the frozen reference vocabulary."""
import json
from pathlib import Path
import numpy as np

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / 'configs/media_catalog.json'


def encode_medium(spec, catalog_path=DEFAULT_CATALOG):
    with open(catalog_path, encoding='utf-8') as handle:
        catalog = json.load(handle)
    components = catalog['component_list']
    if len(components) != 159 or len(set(components)) != 159:
        raise ValueError('Reference medium vocabulary must have 159 unique components')
    if isinstance(spec, str):
        spec = {'medium_id': spec}
    if not isinstance(spec, dict) or not isinstance(spec.get('medium_id'), str) or not spec['medium_id']:
        raise ValueError('Medium requires a nonempty medium_id')
    name = spec['medium_id']
    if 'vector' in spec and 'composition' in spec:
        raise ValueError('Supply a composition or a vector, not both')
    if 'vector' in spec:
        vector = np.asarray(spec['vector'], dtype=np.float32)
    elif 'composition' in spec:
        composition = spec['composition']
        if not isinstance(composition, dict) or not composition:
            raise ValueError('composition must be a nonempty ingredient-to-amount object')
        unknown = set(composition) - set(components)
        if unknown:
            raise ValueError(f'Unknown ingredients: {sorted(unknown)}; use exact names in the reference vocabulary')
        vector = np.array([composition.get(c, 0.0) for c in components], dtype=np.float32)
    elif name in catalog.get('ambiguous_names', {}):
        raise ValueError(f'Medium {name!r} has several reference recipes; provide its composition explicitly')
    elif name in catalog['media']:
        vector = np.asarray(catalog['media'][name], dtype=np.float32)
    else:
        raise ValueError(f'Unknown medium {name!r}; provide composition using the reference ingredient names')
    if vector.shape != (159,) or not np.isfinite(vector).all() or (vector < 0).any():
        raise ValueError('Medium must encode to 159 finite, nonnegative concentrations')
    return {'medium_id': name, 'vector': vector.tolist(), 'params': spec.get('params', {})}
