"""CPU checks of portability, alignment, leakage guards and the actual model stack."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['ACTEXPR_DEVICE'] = 'cpu'
os.environ['ACTEXPR_THREADS'] = '1'
import numpy as np
import torch
torch.set_num_threads(1)
import data
import expression_model as model
import predict
import schema
import split

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rng = np.random.default_rng(17)
    gwd, esm, glm, cls, order = {}, {}, {}, {}, {}
    sets = {'train': ['train'], 'val': ['val'], 'test': ['test']}
    for sid, n in [('train', 100), ('val', 24), ('test', 24)]:
        ids = [f'g{i}' for i in range(n)]
        esm[sid] = rng.normal(size=(n, 480)).astype('float32')
        glm[sid] = rng.normal(size=(n, 960)).astype('float32')
        cls[sid] = rng.normal(size=480).astype('float32')
        order[sid] = ids
        genes = {}
        for i, gid in enumerate(ids):
            conditions = []
            for mid in range(2):
                media = np.zeros(159, dtype='float32'); media[mid] = 1
                conditions.append({'media_name': str(mid), 'media_vector': media,
                                   'expression': float(esm[sid][i, 0] + mid * 0.3)})
            genes[gid] = {'gene_info': {'gene_sequence_dna': 'ACGT' * (4 + i % 3)},
                          'conditions': conditions}
        gwd[sid] = genes
    return sets, {'gene_wise_data': gwd, 'species_metadata': {}}, esm, order, glm, cls, order


class ReleaseTests(unittest.TestCase):
    def test_paths_resolve_relative_to_config_and_not_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'paths.json'
            path.write_text(json.dumps({'DATA_DIR': 'references', 'ANTISMASH_DIR': None}))
            env = dict(os.environ, ACTEXPR_CONFIG=str(path), PYTHONPATH=str(ROOT))
            result = subprocess.check_output([sys.executable, '-c',
                'import config; print(config.DATA_DIR); print(config.ANTISMASH_DIR)'], cwd='/tmp', env=env, text=True)
            self.assertEqual(result.splitlines(), [str(Path(folder)/'references'), 'None'])

    def test_schema_roundtrip_and_dna_features(self):
        ds = schema.Dataset([schema.Genome('G', 'Streptomyces example', [schema.Gene('g', 'MA', 'ACGT')])],
            [schema.Medium('M', vector=[0.0]*159)], [schema.ExpressionRecord('G', 'g', 'M', 2.5)])
        rebuilt = data.to_schema(data.from_schema(ds))
        self.assertEqual(rebuilt.expression, ds.expression)
        with tempfile.TemporaryDirectory() as folder:
            schema.save_dataset(ds, folder)
            self.assertEqual(schema.load_dataset(folder), ds)
        feature = model.dnasum('ACGTN')
        self.assertAlmostEqual(float(feature[0]), 0.5)
        self.assertAlmostEqual(float(feature[1:].sum()), 1.0)

    def test_split_has_no_genus_or_genome_overlap(self):
        tax = {g: split.SpeciesInfo(g, 'schema', g+' example', g, g) for g in split.ORDER}
        media = {g: {'M'} for g in tax}
        fold = split.make_split(tax, media)
        split._assert_no_leakage(fold, tax)
        self.assertIn('Streptomyces', fold.train_species)
        bad = split.Fold('bad', ('Streptomyces',), (), ('Streptomyces',))
        with self.assertRaises(AssertionError):
            split._assert_no_leakage(bad, tax)

    def test_training_uses_explicit_taxonomy_recovery_setting(self):
        ref = {'unified_data': {'gene_wise_data': {'A': {'g': {'conditions': []}}}},
               'esm_emb': {'A': []}, 'glm_emb': {'A': []}}
        cfg = type('Cfg', (), {'DTU_META': 'dtu', 'MANUAL_META': 'manual', 'ANTISMASH_DIR': None})
        fold = split.Fold('test', ('A',), (), ())
        with patch.object(split, 'build_taxonomy', return_value={}) as build, patch.object(split, 'make_split', return_value=fold):
            self.assertEqual(data.reference_split(ref, cfg)['train'], ['A'])
            self.assertIsNone(build.call_args.args[-1])

    def test_query_rejects_collision_and_missing_order(self):
        genome = schema.Genome('new', 'Example species', [schema.Gene('a', 'M', 'ATG'), schema.Gene('b', 'M', 'ATG')])
        medium = schema.Medium('M', vector=[0.0]*159)
        embeddings = {'esm_emb': {'new': np.zeros((2, 480))}, 'esm_order': {'new': ['b', 'a']},
                      'glm_emb': {'new': np.zeros((2, 960))}, 'glm_order': {'new': ['a', 'b']},
                      'glm_cls': {'new': np.zeros(480)}}
        reference = set()
        predict._validate_query(genome, medium, embeddings, reference)
        with self.assertRaises(ValueError):
            predict._validate_query(genome, medium, {k:v for k,v in embeddings.items() if k != 'esm_order'}, reference)
        reference.add('new')
        with self.assertRaises(ValueError):
            predict._validate_query(genome, medium, embeddings, reference)

    def test_real_stack_alignment_and_test_label_isolation(self):
        args = fixture()
        cache = model.build_cache(*args)
        bundle = model.derive_bundle(cache)
        # Permuting gLM rows together with their own gene order must preserve predictions.
        reordered = list(args)
        reordered[4] = {k: v[::-1].copy() for k, v in args[4].items()}
        reordered[6] = {k: list(reversed(v)) for k, v in args[6].items()}
        alternate = model.build_cache(*reordered)
        torch.testing.assert_close(cache['GIOn'], alternate['GIOn'])
        torch.testing.assert_close(cache['NBR_F'], alternate['NBR_F'])
        # Hidden test labels must have no effect on retrieval, validation selection or fitting.
        alternate['Y']['test'] = alternate['Y']['test'] * -100 + 10000
        alternate_bundle = model.derive_bundle(alternate)
        self.assertEqual(bundle['kT'], alternate_bundle['kT'])
        for key in ['GTAB', 'retr_full']:
            torch.testing.assert_close(bundle[key], alternate_bundle[key])
        hp = {'hid': 16, 'nb': 1, 'dm': 8, 'epochs': 2, 'patience': 2, 'seeds': 1, 'bs': 64}
        result = model.train_head(bundle, hp=hp)
        other = model.train_head(alternate_bundle, hp=hp)
        self.assertEqual(result['pred_test'].shape, (48,))
        self.assertTrue(np.isfinite(result['pred_test']).all())
        np.testing.assert_allclose(result['pred_test'], other['pred_test'], rtol=1e-5, atol=1e-6)

    def test_glm_and_preprocessing_imports_do_not_start_jobs(self):
        for name in ['glm.model', 'glm.pretrain', 'glm.extract', 'preprocessing.build_dataset',
                     'preprocessing.genome_features', 'preprocessing.esm_embeddings']:
            importlib.import_module(name)


if __name__ == '__main__':
    unittest.main()
