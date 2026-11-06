"""Raw GenBank/FASTA + medium -> ready-to-use expression-model query files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def prepare(args):
    if args.predict:
        from model_artifacts import check_model
        check_model(args.model_dir)
    import numpy as np
    import torch
    from preprocessing.raw_genome import read_genome, cds_metadata
    from preprocessing.operons import infer_operons
    from preprocessing.query_features import build_features
    from preprocessing.protein_embeddings import embed_proteins, ESM_NAME
    from preprocessing.media import encode_medium
    from glm.extract import extract_one
    source = Path(args.genbank or args.fasta).expanduser().resolve()
    destination = Path(args.out).expanduser().resolve()
    if destination.exists():
        raise ValueError(f'Output already exists; choose a new --out: {destination}')
    for path in [source, Path(args.glm_checkpoint), Path(args.emb_stats), Path(args.media_catalog)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_digest = sha256(source)
    if args.genome_id is None:
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', source.stem).strip('._-') or 'genome'
        # Namespace query IDs separately from reference accessions. The digest
        # distinguishes different inputs that happen to have the same filename.
        args.genome_id = f'query_{stem[:80]}_{input_digest[:12]}'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.genome_id) or args.genome_id in ['.', '..']:
        raise ValueError('genome-id must use letters, digits, dot, underscore or hyphen')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    if args.threads < 1:
        raise ValueError('--threads must be at least 1')
    torch.set_num_threads(args.threads)
    spec = args.medium
    if args.medium_json:
        with open(args.medium_json, encoding='utf-8') as handle:
            spec = json.load(handle)
    medium = encode_medium(spec, args.media_catalog)
    print('[1/5] Parse CDSs / call genes', flush=True)
    records, genome, skipped = read_genome(source, args.genome_id,
        format='genbank' if args.genbank else 'fasta', organism=args.organism,
        topology=args.topology, gene_caller_mode=args.gene_caller_mode)
    if len(genome['genes']) > 13000:
        raise ValueError('Final gLM supports at most 13000 genes; query will not be silently truncated')
    print('[2/5] Infer operons and extract promoter/topology features', flush=True)
    operons, op_summary = infer_operons(genome, mode=args.operon_mode)
    if len(operons) > 9000:
        raise ValueError('Final gLM supports at most 9000 operons')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete bundle atomically; failed runs never appear as a valid query.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=destination.parent) as temporary:
        stage = Path(temporary)/'bundle'; stage.mkdir()
        features = stage/'features'/args.genome_id; features.mkdir(parents=True)
        metadata = cds_metadata(genome)
        write_json(features/'cds_meta.json', metadata)
        feature_info = build_features(genome, records, operons, features, args.unknown_topology)
        write_json(stage/'genome.json', genome)
        write_json(stage/'medium.json', medium)
        print('[3/5] Generate ESM2-35M protein embeddings', flush=True)
        embeddings = embed_proteins([g['aa_sequence'] for g in genome['genes']], device=args.device,
            batch_size=args.batch_size, checkpoint=args.esm_checkpoint)
        np.save(features/'cds_emb.npy', embeddings)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        print('[4/5] Extract final BGC-free gLM embeddings', flush=True)
        arrays = extract_one(features, args.glm_checkpoint, args.emb_stats, device=args.device)
        np.savez_compressed(stage/'query.npz', **arrays)
        print('[5/5] Validate and write query bundle', flush=True)
        if arrays['glm_emb'].shape != (len(genome['genes']), 960):
            raise ValueError('gLM output does not match query genes')
        manifest = {'format_version': 1, 'genome_id': args.genome_id, 'input_format': 'genbank' if args.genbank else 'fasta',
            'input_sha256': input_digest, 'skipped_CDS': skipped,
            'gene_caller': 'existing GenBank CDSs' if args.genbank else f'pyrodigal/{args.gene_caller_mode}',
            'features': feature_info, 'operon_inference': op_summary,
            'esm_model': ESM_NAME, 'esm_layer': 12, 'esm_storage_dtype': 'float16',
            'esm_compute_dtype': 'float16' if args.device.startswith('cuda') else 'float32',
            'protein_crop': 'first 511 + last 511 residues if longer than 1022',
            'glm_checkpoint_sha256': sha256(args.glm_checkpoint), 'glm_cis_dim': 96, 'glm_cis_dil': 4,
            'normalization_sha256': sha256(args.emb_stats), 'media_catalog_sha256': sha256(args.media_catalog),
            'medium_id': medium['medium_id'], 'device': args.device}
        if args.esm_checkpoint:
            manifest['esm_checkpoint_sha256'] = sha256(args.esm_checkpoint)
        if args.predict:
            os.environ['ACTEXPR_DEVICE'] = args.device
            os.environ['ACTEXPR_THREADS'] = str(args.threads)
            from predict import predict
            from schema import Genome, Gene, Medium
            preds = predict(Genome(genome_id=genome['genome_id'], organism=genome['organism'],
                                  genes=[Gene(**g) for g in genome['genes']]), Medium(**medium),
                            {k: {args.genome_id: v.tolist() if k.endswith('_order') else v} for k,v in arrays.items()},
                            model_dir=args.model_dir, device=args.device)
            write_json(stage/'predictions.json', preds)
        write_json(stage/'manifest.json', manifest)
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    print(f'Ready: {destination} ({len(genome["genes"])} genes, {len(operons)} operons)', flush=True)
    return destination


def main(argv=None):
    from preprocessing.media import DEFAULT_CATALOG
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--genbank', help='Annotated nucleotide GenBank (.gbk/.gbff)')
    inputs.add_argument('--fasta', help='Raw nucleotide FASTA; calls CDSs using Pyrodigal')
    parser.add_argument('--genome-id', help='Optional internal ID override; generated automatically from the input file by default')
    parser.add_argument('--organism', help='Override organism name')
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument('--medium', help='Unambiguous named reference medium, e.g. ISP2')
    media.add_argument('--medium-json', help='JSON with medium_id and composition or vector')
    parser.add_argument('--out', required=True, help='New output directory')
    parser.add_argument('--device', default=os.environ.get('ACTEXPR_DEVICE', 'cpu'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--operon-mode', choices=['uniop','singletons'], default='uniop')
    parser.add_argument('--topology', choices=['linear','circular'], help='Required for FASTA; overrides GenBank topology if given')
    parser.add_argument('--unknown-topology', choices=['linear','circular'], default='circular', help='Unknown GenBank topology fallback (matches source recipe)')
    parser.add_argument('--gene-caller-mode', choices=['meta','single'], default='meta')
    parser.add_argument('--esm-checkpoint', help='Optional local ESM2-35M weights (contact-regression file alongside)')
    parser.add_argument('--glm-checkpoint', default=config.GLM_CHECKPOINT)
    parser.add_argument('--emb-stats', default=config.EMB_STATS)
    parser.add_argument('--media-catalog', default=str(DEFAULT_CATALOG))
    parser.add_argument('--predict', action='store_true', help='Also predict using a previously exported expression model; never trains')
    parser.add_argument('--model-dir', default=config.MODEL_DIR, help='Fitted expression model for --predict')
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
