"""Create a code-only ZIP, excluding local configuration, datasets and weights."""
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'.gitignore', 'README.md', 'requirements.txt', 'requirements-tested.txt',
              'config.example.json', 'config.py', 'data.py', 'schema.py', 'split.py',
              'expression_model.py', 'serving.py', 'model_artifacts.py', 'train.py', 'predict.py', 'prepare_inputs.py', 'requirements-preprocessing.txt'}
DIRECTORIES = {'configs', 'docs', 'results/reference', 'glm', 'preprocessing', 'tests', 'examples', 'scripts', '.github'}
EXTENSIONS = {'.py', '.md', '.json', '.txt', '.yaml', '.yml', '.gbk', '.fna', '.fasta'}


def source_files():
    files = {ROOT / name for name in ROOT_FILES}
    for directory in DIRECTORIES:
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and p.suffix in EXTENSIONS and '__pycache__' not in p.parts and '.local.' not in p.name:
                files.add(p)
    for path in sorted(files):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing file or nonportable symlink: {path}')
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'Large artifact in source tree: {path}')
        if re.search(r'/(?:home|Users)/[\w.-]+/', path.read_text(encoding='utf-8')):
            raise ValueError(f'Machine-specific path in public source: {path}')
        yield path


def main():
    files = list(source_files())
    output = ROOT / 'dist'; output.mkdir(exist_ok=True)
    target = output / 'Final_release-source.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        checksums = {}
        for p in files:
            relative = p.relative_to(ROOT).as_posix()
            archive.write(p, 'Final_release/' + relative)
            checksums[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        archive.writestr('Final_release/SHA256SUMS.json', json.dumps(checksums, indent=2) + '\n')
    print(f'{target.name}: {len(files)} source files, {target.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
