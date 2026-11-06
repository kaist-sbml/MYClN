"""One-time training, evaluation and export of the complete expression model."""
import argparse
import json
from pathlib import Path
import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check paths without loading large files')
    parser.add_argument('--output', default=config.MODEL_DIR, help='fitted model directory (also stores evaluation results)')
    parser.add_argument('--hp', default=config.OPTUNA_HP, help='JSON containing an hp object')
    parser.add_argument('--overwrite', action='store_true', help='rebuild an existing model directory')
    args = parser.parse_args(argv)
    if args.check:
        return int(bool(config.check()))
    missing = config.check()
    if missing:
        parser.error('missing required artifacts: ' + ', '.join(missing))
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
        parser.error('output is not empty; use another --output or --overwrite')
    with open(args.hp, encoding='utf-8') as handle:
        hp = json.load(handle)['hp']
    output.mkdir(parents=True, exist_ok=True)
    # A cancelled/failed rebuild must never be mistaken for a complete model.
    (output / 'manifest.json').unlink(missing_ok=True)
    print(f'[1/4] Loading reference data; one-time model export to {output}', flush=True)
    import numpy as np
    import data
    import expression_model as model
    ref = data.load_reference(config)
    sets = data.reference_split(ref, config)
    print('[split]', {k: len(v) for k, v in sets.items()}, flush=True)
    print('[2/4] Building and saving reference indexes', flush=True)
    cache = model.build_cache(sets, ref['unified_data'], ref['esm_emb'], ref['esm_order'],
                              ref['glm_emb'], ref['glm_cls'], ref['glm_order'], artifact_dir=output)
    bundle = model.derive_bundle(cache)
    print('[3/4] Training and saving heads / LightGBM / ridge', flush=True)
    result = model.train_head(bundle, hp=hp, artifact_dir=output)
    inv = {i: key for key, i in cache['gidx'].items()}
    keys = [inv[i] for i in result['gene_idx_test']]
    output.mkdir(parents=True, exist_ok=True)
    metrics = {'test_pearson_r': float(result['test_r']),
               'within_condition_r': float(result['within_condition_r']),
               'n_test': int(len(result['pred_test'])), 'split': sets, 'head_hp': hp}
    with (output / 'result.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(output / 'test_predictions.npz',
        species=np.array([k[0] for k in keys]), gene=np.array([k[1] for k in keys]),
        condition=cache['EXP']['test'], pred=result['pred_test'], measured=result['y_test'])
    print(f"test Pearson r={result['test_r']:.4f}; within-condition r={result['within_condition_r']:.4f}")
    from model_artifacts import finalize_model
    finalize_model(output)
    print(f'[4/4] Model ready: {output}. Prediction now loads this model without training.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
