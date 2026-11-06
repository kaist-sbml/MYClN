"""Portable artifact paths. Override with ACTEXPR_CONFIG or ACTEXPR_<NAME>.

JSON paths are relative to the JSON file; defaults are relative to this repository.
Importing this module never loads data, starts training, or creates directories.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_config_file = os.environ.get('ACTEXPR_CONFIG')
if _config_file:
    _config_path = Path(_config_file).expanduser().resolve()
    with _config_path.open(encoding='utf-8') as handle:
        _overrides = json.load(handle)
    if not isinstance(_overrides, dict):
        raise ValueError('ACTEXPR_CONFIG must contain a JSON object')
    _base = _config_path.parent
else:
    _overrides, _base = {}, ROOT


def path(name, default):
    """Resolve one configurable path; null disables an optional input."""
    value = os.environ.get('ACTEXPR_' + name, _overrides.get(name, default))
    if value is None or value == '':
        return None
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else _base / p)


DATA_DIR = path('DATA_DIR', str(ROOT / 'artifacts/reference'))
EMB_DIR = path('EMB_DIR', str(Path(DATA_DIR) / 'embeddings'))
RAW_DATA_DIR = path('RAW_DATA_DIR', str(ROOT / 'data/raw'))
TOOLS_DIR = path('TOOLS_DIR', str(ROOT / 'data/tools'))
DATASET_DIR = path('DATASET_DIR', str(ROOT / 'outputs/dataset'))
MODEL_DIR = path('MODEL_DIR', str(ROOT / 'artifacts/expression'))
UNIFIED_DATA = path('UNIFIED_DATA', str(Path(DATA_DIR) / 'unified_expression_data.pkl'))
ESM_EMB = path('ESM_EMB', str(Path(EMB_DIR) / 'bert_gene_raw_v5.pkl'))
ESM_ORDER = path('ESM_ORDER', str(Path(EMB_DIR) / 'bert_gene_id_order_e2e.pkl'))
GLM_EMB = path('GLM_EMB', str(Path(EMB_DIR) / 'bert_gene_embs_v7bgcoff.pkl'))
GLM_CLS = path('GLM_CLS', str(Path(EMB_DIR) / 'bert_cls_vectors_v7bgcoff.pkl'))
GLM_ORDER = path('GLM_ORDER', str(Path(EMB_DIR) / 'bert_gene_id_order_v7bgcoff.pkl'))
DTU_META = path('DTU_META', str(Path(RAW_DATA_DIR) / 'DTU_Pep_data/DTU_Pep_metadata.csv'))
MANUAL_META = path('MANUAL_META', str(Path(RAW_DATA_DIR) / 'Manual_curated_data/Metadata/Actinomycetes_metadata_20260106.csv'))
# Keep taxonomy recovery OFF for the historical canonical benchmark.
ANTISMASH_DIR = path('ANTISMASH_DIR', None)
OPTUNA_HP = path('OPTUNA_HP', str(ROOT / 'configs/head.json'))
GLM_CHECKPOINT = path('GLM_CHECKPOINT', str(ROOT / 'artifacts/glm/v7_cisfull_bgcoff_best.pth'))
EMB_STATS = path('EMB_STATS', str(ROOT / 'artifacts/glm/v5_emb_stats.npz'))
GLM_WARM_START = path('GLM_WARM_START', str(ROOT / 'artifacts/glm/v6_topo_scratch_bgcoff_best.pth'))
GENOME_CORPUS = path('GENOME_CORPUS', str(Path(RAW_DATA_DIR) / 'Whole_genome_data/cds_embeddings'))
GLM_OUTPUT_DIR = path('GLM_OUTPUT_DIR', str(ROOT / 'outputs/glm'))
EXTRACT_OUTPUT_DIR = path('EXTRACT_OUTPUT_DIR', str(ROOT / 'outputs/embeddings'))
SPECIES_ORDER = path('SPECIES_ORDER', ESM_ORDER)
GENOME_ROOTS = [
    path('DTU_GENOME_ROOT', str(Path(RAW_DATA_DIR) / 'DTU_Pep_data/Whole_genome_inputs/cds_embeddings2')),
    GENOME_CORPUS,
    path('MANUAL_GENOME_ROOT', str(Path(RAW_DATA_DIR) / 'Manual_curated_data/Genome/cds_embeddings')),
]
REFERENCE_FILES = ('UNIFIED_DATA', 'ESM_EMB', 'ESM_ORDER', 'GLM_EMB', 'GLM_CLS', 'GLM_ORDER')


def check(scope='train'):
    names = list(REFERENCE_FILES) + ['DTU_META', 'MANUAL_META', 'OPTUNA_HP'] if scope == 'train' else [
        'GLM_CHECKPOINT', 'EMB_STATS', 'SPECIES_ORDER']
    missing = []
    for name in names:
        value = globals()[name]
        exists = value is not None and Path(value).is_file()
        print(f'{"OK" if exists else "MISSING":7} {name}: {value}')
        if not exists:
            missing.append(name)
    return missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['train', 'glm'], default='train')
    args = parser.parse_args()
    raise SystemExit(bool(check(args.scope)))
