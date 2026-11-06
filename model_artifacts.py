"""Small, dependency-free checks for a complete fitted expression model."""
import json
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
BASE_FILES = (
    'keys_full.npy', 'keys_dual.npy', 'keys_raw.npy', 'keys_gio.npy',
    'feature_stats.npz', 'train_mean.npy', 'expression.npy', 'has_expression.npy',
    'media_vectors.npy', 'reference_metadata.json', 'fit_metadata.json',
    'transforms.npz', 'stack.npz', 'lightgbm.txt',
)


def required_files(directory):
    fit = json.loads((directory / 'fit_metadata.json').read_text(encoding='utf-8'))
    seeds = fit.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Invalid fitted head list')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate fitted heads')
    return (*BASE_FILES, *(f'heads/seed_{seed}.pt' for seed in seeds))


def finalize_model(directory):
    """Publish the ready marker LAST, after all training outputs have been saved."""
    directory = Path(directory)
    files = {}
    for name in required_files(directory):
        size = (directory / name).stat().st_size
        if not size:
            raise ValueError(f'Empty model file: {name}')
        files[name] = size
    manifest = {'format_version': FORMAT_VERSION, 'status': 'complete',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files}
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(directory / 'manifest.json')
    return manifest


def check_model(directory):
    """Check completeness without importing torch or loading the reference arrays."""
    directory = Path(directory).expanduser().resolve()
    marker = directory / 'manifest.json'
    if not marker.is_file():
        raise FileNotFoundError(
            f'No complete fitted expression model at {directory}. '
            f'Run train.py --output "{directory}" once, then retry prediction. '
            'Prediction never trains a model.')
    try:
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest.get('format_version') != FORMAT_VERSION or manifest.get('status') != 'complete':
            raise ValueError('unsupported model format or incomplete export')
        expected = required_files(directory)
        if set(manifest.get('files', {})) != set(expected):
            raise ValueError('model file list does not match fitted heads')
        for name in expected:
            if not (directory / name).is_file() or (directory / name).stat().st_size != manifest['files'][name]:
                raise ValueError(f'missing or truncated model file: {name}')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f'Invalid expression model at {directory}: {exc}') from exc
    return directory
