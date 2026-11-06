"""Prepare the Hugging Face model card and checksums locally; never uploads."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_model import ROOT, bundle_files, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-dir', type=Path, default=ROOT/'artifacts')
    args = parser.parse_args(argv)
    directory = args.artifacts_dir.resolve()
    files = bundle_files(directory/'expression', directory/'glm')
    # Names intentionally include artifacts/: verify from the source checkout root.
    checksums = []
    for source, name in files:
        print(f'Hashing {name}', flush=True)
        checksums.append(f'{sha256(source)}  {name}\n')
    temporary = directory/'SHA256SUMS.tmp'
    temporary.write_text(''.join(checksums), encoding='utf-8')
    temporary.replace(directory/'SHA256SUMS')
    card = directory/'README.md'
    if not card.exists():
        card.write_text((ROOT/'docs/HF_MODEL_CARD.md').read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Ready: {len(files)} model files, {sum(p.stat().st_size for p, _ in files):,} bytes')
    print(f'Model card: {card}; checksums: {directory / "SHA256SUMS"}')
    print('No repository was created and no files were uploaded.')


if __name__ == '__main__':
    main()
