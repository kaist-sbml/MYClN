"""Create deterministic synthetic GenBank/FASTA examples for pipeline smoke checks."""
from pathlib import Path
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation


def create_examples(output):
    rng = random.Random(41)
    codons = ['GCT','GCC','GGT','GGC','CTG','GAG','GAC','AAC','CAG','CCG','ACG','GTG','AAG','CGC','TTC','ATC']
    seq = 'ACGT' * 80
    features = []
    for i in range(64):
        strand = 1 if i % 4 < 2 else -1
        dna = 'ATG' + ''.join(rng.choice(codons) for _ in range(59)) + 'TAA'
        start = len(seq); aa = str(Seq(dna).translate())[:-1]
        seq += dna if strand == 1 else str(Seq(dna).reverse_complement())
        features.append(SeqFeature(FeatureLocation(start, len(seq), strand=strand), type='CDS',
            qualifiers={'locus_tag': [f'demo_{i+1:04d}'], 'translation': [aa], 'transl_table': ['11']}))
        gap = 20 if i % 2 == 0 else 180
        seq += ''.join(rng.choice('ACGT') for _ in range(gap))
    seq += 'ACGT' * 80
    record = SeqRecord(Seq(seq), id='synthetic_contig', name='synthetic_contig',
        description='Synthetic coding sequence example; no biological accuracy claims',
        annotations={'molecule_type':'DNA','topology':'linear','organism':'Synthetic example'}, features=features)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    SeqIO.write([record], output/'example.gbk', 'genbank')
    SeqIO.write([record], output/'example.fna', 'fasta')


if __name__ == '__main__':
    create_examples(Path(__file__).resolve().parents[1]/'examples')
