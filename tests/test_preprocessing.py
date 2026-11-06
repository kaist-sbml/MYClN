"""Checks for raw-input preparation without downloading model weights."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from preprocessing.raw_genome import read_genome, cds_metadata
from preprocessing.operons import infer_operons
from preprocessing.query_features import build_features
from preprocessing.protein_embeddings import clean_protein
from preprocessing.media import encode_medium, DEFAULT_CATALOG
from preprocessing import genome_features as legacy

ROOT = Path(__file__).resolve().parents[1]


class RawInputTests(unittest.TestCase):
    def example(self):
        return read_genome(ROOT/'examples/example.gbk', 'query')

    def test_genbank_cds_and_reverse_strand(self):
        records, genome, skipped = self.example()
        self.assertEqual(len(genome['genes']), 64)
        self.assertFalse(skipped)
        for gene in genome['genes']:
            self.assertEqual(gene['aa_sequence'], str(Seq(gene['dna_sequence']).translate())[:-1])
        self.assertEqual(genome['genes'][2]['strand'], -1)
        self.assertEqual(genome['genes'][2]['dna_sequence'][:3], 'ATG')

    def test_fasta_gene_calling(self):
        try:
            import pyrodigal
        except ImportError:
            self.skipTest('Install requirements-preprocessing.txt to test FASTA gene calling')
        records, genome, _ = read_genome(ROOT/'examples/example.fna', 'query', format='fasta', topology='linear')
        self.assertGreaterEqual(len(genome['genes']), 2)
        self.assertTrue(all(g['aa_sequence'] and g['dna_sequence'] for g in genome['genes']))
        self.assertEqual(records[0].annotations['topology'], 'linear')

    def test_fasta_requires_topology(self):
        with self.assertRaisesRegex(ValueError, 'topology'):
            read_genome(ROOT/'examples/example.fna', 'query', format='fasta')

    def test_uniop_is_deterministic_and_complete(self):
        _, genome, _ = self.example()
        a, sa = infer_operons(genome)
        b, sb = infer_operons(genome)
        self.assertEqual(a, b); self.assertEqual(sa, sb)
        members = [gid for op in a for gid in op['genes']]
        self.assertEqual(sorted(members), sorted(g['gene_id'] for g in genome['genes']))
        self.assertLess(len(a), len(genome['genes']))
        short = {**genome, 'genes': genome['genes'][:4]}
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            infer_operons(short)
        singletons, summary = infer_operons(short, mode='singletons')
        self.assertEqual(len(singletons), 4)
        self.assertEqual(summary['mode'], 'singletons')

    def test_query_features_match_original_feature_producer(self):
        records, genome, _ = self.example()
        ops, _ = infer_operons(genome)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'new'/'query'; path.mkdir(parents=True)
            old = Path(folder)/'old'/'query'; old.mkdir(parents=True)
            metadata = cds_metadata(genome)
            for dest in [path, old]:
                (dest/'cds_meta.json').write_text(json.dumps(metadata))
                np.save(dest/'cds_emb.npy', np.zeros((len(metadata), 480), np.float16))
            info = build_features(genome, records, ops, path)
            op_map, starts = legacy.load_operon_table(path/'operons.tsv')
            promoters = legacy.load_promoter_fasta(path/'promoters.fasta')
            legacy._init_worker(op_map, starts, promoters)
            status, _, error = legacy.process_genome(str(old))
            self.assertEqual(status, 'success', error)
            with np.load(path/'operon_meta.npz') as a, np.load(old/'operon_meta.npz') as b:
                self.assertEqual(set(a.files), set(b.files))
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            with np.load(path/'topology_meta.npz') as topo:
                self.assertTrue((topo['operon_topology'] == 0).all())
                self.assertTrue((topo['operon_contig_start'] == 0).all())
                self.assertTrue((topo['operon_contig_len'] == len(ops)).all())
            self.assertEqual(info['unknown_topology_contigs'], [])

    def test_named_medium_matches_composition(self):
        with open(ROOT/'examples/medium_composition.json') as handle:
            spec = json.load(handle)
        a = encode_medium('ISP2'); b = encode_medium(spec)
        self.assertEqual(a['vector'], b['vector'])
        self.assertEqual(len(a['vector']),159)

    def test_media_failures_are_explicit(self):
        for spec in ['not_a_medium', 'BHI', {'medium_id':'x','composition':{'Glucose typo':1}},
                     {'medium_id':'x','composition':{'Glucose':-1}},
                     {'medium_id':'x','vector':[0]*158}]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                encode_medium(spec)

    def test_esm_crop_and_clean_match_training_recipe(self):
        self.assertEqual(clean_protein('abjz*'), 'AXXXX')
        long = 'A'*600+'G'*600
        self.assertEqual(clean_protein(long), 'A'*511+'G'*511)


if __name__ == '__main__':
    unittest.main()
