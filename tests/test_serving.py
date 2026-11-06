"""Persistence equivalence and independence of serving from training inputs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_release import fixture
import expression_model as model
from model_artifacts import check_model, finalize_model
from predict import predict
from schema import Gene, Genome, Medium
from serving import ExpressionPredictor

ROOT = Path(__file__).resolve().parents[1]
HP = {'hid': 16, 'nb': 1, 'dm': 8, 'epochs': 2, 'patience': 2, 'seeds': 2, 'bs': 64}


def query(args, mid=0):
    genes = args[1]['gene_wise_data']['test']
    genome = Genome('test', 'Synthetic test organism', [
        Gene(gid, 'MA', gd['gene_info']['gene_sequence_dna']) for gid, gd in genes.items()])
    vector = np.zeros(159, dtype=np.float32); vector[mid] = 1
    medium = Medium(str(mid), vector=vector.tolist())
    embeddings = {name: {'test': table['test']} for name, table in zip(
        ['esm_emb', 'esm_order', 'glm_emb', 'glm_cls', 'glm_order'], args[2:])}
    return genome, medium, embeddings


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name) / 'model'
        cls.args = fixture()
        # Two reference genomes give the genome-CLS channel nonzero train
        # variance, as in the real corpus (a one-genome fixture overflows fp16).
        sets, uni, esm, order, glm, genome_cls, glm_order = cls.args
        sets['train'].append('train2')
        items = list(uni['gene_wise_data']['train'].items())
        uni['gene_wise_data']['train'] = dict(items[:50])
        uni['gene_wise_data']['train2'] = dict(items[50:])
        for table in (esm, glm):
            table['train2'] = table['train'][50:].copy()
            table['train'] = table['train'][:50].copy()
        order['train2'] = order['train'][50:]
        order['train'] = order['train'][:50]
        genome_cls['train2'] = -genome_cls['train']
        cache = model.build_cache(*cls.args, artifact_dir=cls.directory)
        cls.bundle = model.derive_bundle(cache)
        cls.expected = model.train_head(cls.bundle, hp=HP, artifact_dir=cls.directory)['pred_test'].reshape(-1, 2)
        finalize_model(cls.directory)
        cls.predictor = ExpressionPredictor(cls.directory, device='cpu', verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_reloaded_model_matches_training_predictions_without_fitting(self):
        with patch('data.load_reference', side_effect=AssertionError('Reference read during inference')), \
             patch.object(model, 'build_cache', side_effect=AssertionError('Cache rebuilt during inference')), \
             patch.object(model, 'train_head', side_effect=AssertionError('Training during inference')), \
             patch.object(torch.optim, 'AdamW', side_effect=AssertionError('Optimizer during inference')):
            for mid in (0, 1):
                actual = self.predictor.predict(*query(self.args, mid))
                np.testing.assert_allclose(list(actual.values()), self.expected[:, mid], rtol=1e-5, atol=1e-5)
            actual = predict(*query(self.args), model_dir=self.directory, device='cpu')
            np.testing.assert_allclose(list(actual.values()), self.expected[:, 0], rtol=1e-5, atol=1e-5)

    def test_query_order_subsets_and_retrieval_tiles_do_not_change_result(self):
        genome, medium, embeddings = query(self.args)
        expected = self.predictor.predict(genome, medium, embeddings)
        reordered = copy.deepcopy(embeddings)
        for channel, order in [('esm_emb', 'esm_order'), ('glm_emb', 'glm_order')]:
            reordered[channel]['test'] = reordered[channel]['test'][::-1].copy()
            reordered[order]['test'] = list(reversed(reordered[order]['test']))
        tiled = ExpressionPredictor(self.directory, device='cpu', query_chunk=3, key_tile=37, verbose=False)
        genome.genes = list(reversed(genome.genes))
        actual = tiled.predict(genome, medium, reordered)
        for gid in expected:
            np.testing.assert_allclose(actual[gid], expected[gid], rtol=1e-5, atol=1e-5)
        genome.genes = genome.genes[:1]
        actual = tiled.predict(genome, medium, reordered)
        np.testing.assert_allclose(actual[genome.genes[0].gene_id], expected[genome.genes[0].gene_id], rtol=1e-5, atol=1e-5)

    def test_unseen_medium_matches_original_zero_coverage_fallback(self):
        alternate = copy.deepcopy(self.args)
        vector = np.zeros(159, dtype=np.float32); vector[:2] = [0.25, 0.75]
        for gd in alternate[1]['gene_wise_data']['test'].values():
            for condition in gd['conditions']:
                condition['media_vector'] = vector
        expected = model.train_head(model.derive_bundle(model.build_cache(*alternate)), hp=HP)['pred_test'][::2]
        genome, medium, embeddings = query(self.args)
        medium.vector = vector.tolist()
        actual = self.predictor.predict(genome, medium, embeddings)
        np.testing.assert_allclose(list(actual.values()), expected, rtol=1e-5, atol=1e-5)

    def test_cli_fresh_process_needs_only_model_and_query(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            genome, medium, embeddings = query(self.args)
            (folder / 'genome.json').write_text(json.dumps(asdict(genome)))
            (folder / 'medium.json').write_text(json.dumps(asdict(medium)))
            np.savez(folder / 'query.npz', **{name: table['test'] for name, table in embeddings.items()})
            cfg = folder / 'config.json'
            cfg.write_text(json.dumps({'DATA_DIR': str(folder / 'no-reference'),
                                       'MODEL_DIR': str(self.directory)}))
            env = dict(os.environ, ACTEXPR_CONFIG=str(cfg), ACTEXPR_DEVICE='cpu')
            result = subprocess.run([sys.executable, str(ROOT / 'examples/predict_json.py'),
                '--genome', str(folder / 'genome.json'), '--medium', str(folder / 'medium.json'),
                '--embeddings', str(folder / 'query.npz'), '--output', str(folder / 'out.json'),
                '--threads', '1'], cwd=folder, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            np.testing.assert_allclose(list(json.loads((folder / 'out.json').read_text()).values()),
                                       self.expected[:, 0], rtol=1e-5, atol=1e-5)

    def test_missing_or_incomplete_model_fails_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, 'never trains'):
                check_model(temp)
            code = '''import sys
from examples.predict_json import main
try:
    main(['--model-dir', sys.argv[1], '--genome', 'absent', '--medium', 'absent',
          '--embeddings', 'absent', '--output', 'absent'])
except SystemExit as exc:
    assert exc.code == 2
    assert 'torch' not in sys.modules
else:
    raise AssertionError('Missing model was accepted')
'''
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, '-c', code, temp], env=env, cwd=temp,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('never trains', result.stderr)
        stack = self.directory / 'stack.npz'
        contents = stack.read_bytes()
        try:
            stack.write_bytes(contents[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                check_model(self.directory)
        finally:
            stack.write_bytes(contents)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_export_and_reload_matches_training(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(model, 'DEVICE', torch.device('cuda')), \
             patch.object(model, 'RETR_DTYPE', torch.float16):
            cache = model.build_cache(*self.args, artifact_dir=temp)
            result = model.train_head(model.derive_bundle(cache), hp=HP, artifact_dir=temp)
            finalize_model(temp)
            serving = ExpressionPredictor(temp, device='cuda', verbose=False)
            for mid in (0, 1):
                actual = serving.predict(*query(self.args, mid))
                # fp16 GEMM with query-only rows can round one ULP differently
                # from the full-cache GEMM, including near-tied neighbor order.
                np.testing.assert_allclose(list(actual.values()), result['pred_test'][mid::2], rtol=5e-4, atol=5e-4)
            # A CUDA-exported artifact also loads on a CPU-only serving device.
            cpu = ExpressionPredictor(temp, device='cpu', verbose=False)
            actual = cpu.predict(*query(self.args))
            self.assertTrue(np.isfinite(list(actual.values())).all())


if __name__ == '__main__':
    unittest.main()


class DistributionTests(unittest.TestCase):
    """Reference-row mapping and the model distribution bundle."""
    def test_reference_gene_map_matches_index_and_bundle_is_complete(self):
        import tarfile
        sys.path.insert(0, str(ROOT / 'scripts'))
        import export_reference_genes, package_model
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); directory = temp / 'model'
            args = fixture()
            sets, uni, esm, order = args[:4]
            cache = model.build_cache(*args, artifact_dir=directory)
            model.train_head(model.derive_bundle(cache), hp=HP, artifact_dir=directory)
            finalize_model(directory)
            saved = [line.rstrip('\n').split('\t')[1:] for line in
                     (directory / 'reference_genes.tsv').read_text().splitlines()[1:]]
            rebuilt = export_reference_genes.reference_order(uni, sets['train'])
            self.assertEqual(saved, [list(k) for k in rebuilt])
            export_reference_genes.check_against_index(rebuilt, directory, {'esm_emb': esm, 'esm_order': order})
            with self.assertRaises(ValueError):
                export_reference_genes.check_against_index(rebuilt[::-1], directory, {'esm_emb': esm, 'esm_order': order})
            (directory / 'result.json').write_text('{}')
            np.savez(directory / 'test_predictions.npz', pred=np.zeros(1))
            glm = temp / 'glm'; glm.mkdir()
            for name in package_model.GLM_FILES:
                (glm / name).write_bytes(b'weights')
            package_model.main(['--model-dir', str(directory), '--glm-dir', str(glm), '--output', str(temp / 'dist')])
            with tarfile.open(temp / 'dist' / 'Final_release-model.tar') as archive:
                names = set(archive.getnames())
            self.assertIn('artifacts/expression/heads/seed_0.pt', names)
            self.assertIn('artifacts/expression/keys_full.npy', names)
            self.assertIn('artifacts/expression/reference_genes.tsv', names)
            self.assertIn('artifacts/glm/v7_cisfull_bgcoff_best.pth', names)
            self.assertIn('artifacts/SHA256SUMS', names)
