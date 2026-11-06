"""Predict with a saved expression model; no reference-data loading or training."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from model_artifacts import check_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genome', required=True, help='Genome JSON with genes list')
    parser.add_argument('--medium', required=True, help='Medium JSON with 159-dimensional vector')
    parser.add_argument('--embeddings', required=True, help='NPZ with esm_emb, esm_order, glm_emb, glm_order, glm_cls')
    parser.add_argument('--output', required=True, help='output JSON, must not already exist')
    parser.add_argument('--model-dir', default=config.MODEL_DIR, help='Model exported once by train.py')
    parser.add_argument('--device', default=os.environ.get('ACTEXPR_DEVICE'), help='cpu, cuda, or cuda:N (default: auto)')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads used during inference')
    args = parser.parse_args(argv)
    try:
        # Fail before importing torch or opening genome/embedding/reference files.
        directory = check_model(args.model_dir)
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError('output already exists; choose a new --output')
        if args.threads < 1:
            raise ValueError('--threads must be positive')
        os.environ['ACTEXPR_THREADS'] = str(args.threads)
        if args.device:
            os.environ['ACTEXPR_DEVICE'] = args.device
        import numpy as np
        import torch
        torch.set_num_threads(args.threads)
        from schema import Gene, Genome, Medium
        from predict import predict
        with open(args.genome, encoding='utf-8') as handle:
            raw = json.load(handle)
        genome = Genome(**{**raw, 'genes': [Gene(**g) for g in raw['genes']]})
        with open(args.medium, encoding='utf-8') as handle:
            medium = Medium(**json.load(handle))
        with np.load(args.embeddings, allow_pickle=False) as arrays:
            embeddings = {name: {genome.genome_id: arrays[name].tolist() if name.endswith('_order') else arrays[name]}
                          for name in ['esm_emb', 'esm_order', 'glm_emb', 'glm_order', 'glm_cls']}
        result = predict(genome, medium, embeddings, model_dir=directory, device=args.device)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(f'Saved: {destination}', flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
