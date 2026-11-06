"""Read annotated GenBank, or call CDSs from nucleotide FASTA with Pyrodigal."""
from collections import Counter
from pathlib import Path
from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.SeqFeature import SeqFeature, FeatureLocation


def read_genome(path, genome_id, *, format='genbank', organism=None,
                topology=None, gene_caller_mode='meta'):
    with Path(path).open(encoding='utf-8') as handle:
        records = list(SeqIO.parse(handle, format))
    if not records or any(not len(r.seq) for r in records):
        raise ValueError('Input must contain nonempty nucleotide records')
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Contig IDs must be unique')
    for r in records:
        if set(str(r.seq).upper()) - set('ACGTRYSWKMBDHVN'):
            raise ValueError(f'{r.id}: expected nucleotide sequence (IUPAC DNA)')
        r.annotations['molecule_type'] = 'DNA'
        if topology:
            r.annotations['topology'] = topology
    if format == 'fasta':
        if topology is None:
            raise ValueError('FASTA requires --topology linear or circular')
        try:
            import pyrodigal
        except ImportError as exc:
            raise ImportError('Install requirements-preprocessing.txt for FASTA gene calling') from exc
        finder = pyrodigal.GeneFinder(meta=gene_caller_mode == 'meta')
        if gene_caller_mode == 'single':
            finder.train(*(bytes(r.seq) for r in records), translation_table=11)
        for r in records:
            for i, g in enumerate(finder.find_genes(bytes(r.seq)), 1):
                # Pyrodigal coordinates are 1-based inclusive; BioPython uses 0-based half-open.
                r.features.append(SeqFeature(FeatureLocation(g.begin - 1, g.end, strand=g.strand),
                    type='CDS', qualifiers={'locus_tag': [f'{r.id}_cds{i:05d}'],
                    'translation': [g.translate(include_stop=False)]}))
    genes, skipped = [], Counter()
    for r in records:
        contig_genes = []
        for i, feat in enumerate(r.features):
            if feat.type != 'CDS':
                continue
            if 'pseudo' in feat.qualifiers or 'pseudogene' in feat.qualifiers:
                skipped['pseudogene'] += 1
                continue
            if feat.location is None or feat.location.strand not in (-1, 1):
                raise ValueError(f'{r.id}: CDS without an unambiguous position/strand')
            parts = list(feat.location.parts)
            if len(parts) > 1:
                raise ValueError(f'{r.id}: compound CDS locations require curated annotation before this pipeline')
            start, end = int(feat.location.start), int(feat.location.end)
            if not 0 <= start < end <= len(r.seq):
                raise ValueError(f'{r.id}: CDS coordinates outside contig')
            q = feat.qualifiers
            gid = next((q[k][0] for k in ['locus_tag', 'protein_id', 'gene'] if q.get(k)),
                       f'{r.id}_cds{i+1:05d}')
            dna = str(feat.extract(r.seq)).upper()
            if q.get('translation'):
                aa = ''.join(q['translation'][0].split()).rstrip('*')
            else:
                if 'transl_except' in q:
                    raise ValueError(f'{gid}: transl_except needs a supplied translation')
                table = int(q.get('transl_table', ['11'])[0])
                offset = int(q.get('codon_start', ['1'])[0]) - 1
                seq = feat.extract(r.seq)[offset:]
                aa = str(seq[:len(seq)//3*3].translate(table=table)).rstrip('*')
                if aa and str(seq[:3]).upper() in CodonTable.unambiguous_dna_by_id[table].start_codons:
                    aa = 'M' + aa[1:]
            if not aa or '*' in aa:
                raise ValueError(f'{gid}: empty translation or internal stop codon; curate CDS annotation')
            contig_genes.append({'gene_id': str(gid), 'aa_sequence': aa, 'dna_sequence': dna,
                'contig_id': r.id, 'strand': int(feat.location.strand), 'start': start, 'end': end})
        genes.extend(sorted(contig_genes, key=lambda g: (g['start'], g['end'], g['gene_id'])))
    if len(genes) < 2:
        raise ValueError('At least two translated CDSs are required')
    duplicates = [gid for gid, n in Counter(g['gene_id'] for g in genes).items() if n > 1]
    if duplicates:
        raise ValueError(f'Duplicate gene IDs across genome: {duplicates[:5]}')
    organism = organism or records[0].annotations.get('organism', '') or genome_id
    return records, {'genome_id': genome_id, 'organism': organism, 'genes': genes}, dict(skipped)


def cds_metadata(genome):
    return [{'contig': g['contig_id'], 'start': g['start'], 'end': g['end'],
             'strand': g['strand'], 'locus_tag': g['gene_id']} for g in genome['genes']]
