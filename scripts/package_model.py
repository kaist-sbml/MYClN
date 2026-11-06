"""Bundle the fitted expression model and gLM weights for distribution.

Writes dist/Final_release-model.tar (uncompressed; fp16/npy arrays barely
compress) and dist/Final_release-model.SHA256SUMS. The archive unpacks to
artifacts/expression/ and artifacts/glm/ inside a source checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_artifacts import check_model, required_files  # noqa: E402

GLM_FILES = ('v7_cisfull_bgcoff_best.pth', 'v5_emb_stats.npz')
EXTRA_FILES = ('reference_genes.tsv', 'manifest.json', 'result.json', 'test_predictions.npz')


def sha256(path, block=1 << 24):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_files(model_dir, glm_dir):
    model_dir = check_model(model_dir)
    names = list(required_files(model_dir)) + list(EXTRA_FILES)
    files = [(model_dir / n, f'artifacts/expression/{n}') for n in names]
    files += [(Path(glm_dir) / n, f'artifacts/glm/{n}') for n in GLM_FILES]
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        raise FileNotFoundError('Missing distribution files: ' + ', '.join(missing))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default=str(ROOT / 'artifacts/expression'))
    parser.add_argument('--glm-dir', default=str(ROOT / 'artifacts/glm'))
    parser.add_argument('--output', default=str(ROOT / 'dist'))
    args = parser.parse_args(argv)
    files = bundle_files(args.model_dir, args.glm_dir)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for src, arcname in files:
        checksums[arcname] = sha256(src)
        print(f'  {arcname}  {src.stat().st_size:,} bytes', flush=True)
    sums = out / 'Final_release-model.SHA256SUMS'
    sums.write_text(''.join(f'{h}  {n}\n' for n, h in checksums.items()), encoding='utf-8')
    target = out / 'Final_release-model.tar'
    with tarfile.open(target, 'w') as archive:
        for src, arcname in files:
            archive.add(src, arcname)
        archive.add(sums, 'artifacts/SHA256SUMS')
    print(f'{target}: {len(files)} files, {target.stat().st_size:,} bytes')
    print(f'Verify after extraction: sha256sum -c artifacts/SHA256SUMS (run from the checkout root)')


if __name__ == '__main__':
    main()
