"""Named recipe corrections and explicit composition input behavior."""
import json
import unittest
from pathlib import Path
import numpy as np
from preprocessing.media import encode_medium, DEFAULT_CATALOG

ROOT = Path(__file__).resolve().parents[1]


class MediumCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(DEFAULT_CATALOG.read_text())
        cls.names = cls.catalog['component_list']

    def sparse(self, name):
        return {k: v for k, v in zip(self.names, encode_medium(name)['vector']) if v}

    def test_corrected_named_recipes(self):
        self.assertEqual(self.sparse('ISP2'), {'Glucose': 4., 'Malt extract': 10., 'Yeast extract': 4.})
        self.assertEqual(self.sparse('M9'), {'Na2HPO4': 33.900001525878906, 'KH2PO4': 15., 'NH4Cl': 5., 'NaCl': 2.5})
        self.assertEqual(self.sparse('TSB50'), {'Trypsin_soy_broth': 15.})
        cgxii = self.sparse('CGXII')
        self.assertEqual(len(cgxii), 14)
        self.assertEqual(cgxii['Xylose'], 20.)
        self.assertNotIn('GABA', cgxii)
        self.assertNotIn('MgCl2·7H2O', self.sparse('Media14'))
        self.assertAlmostEqual(self.sparse('Media14')['MgCl2·6H2O'], .55, places=6)

    def test_supplied_aliases(self):
        for alias, target in self.catalog['component_aliases'].items():
            with self.subTest(alias=alias):
                a = encode_medium({'medium_id': 'custom', 'composition': {alias: 2.5}})
                b = encode_medium({'medium_id': 'custom', 'composition': {target: 2.5}})
                self.assertEqual(a['vector'], b['vector'])
        with self.assertRaisesRegex(ValueError, 'Duplicate ingredient'):
            encode_medium({'medium_id': 'x', 'composition': {'Casamino acids': 1, 'Casamino acid': 2}})

    def test_undefined_requires_explicit_recipe(self):
        for name in ['BHI', 'HIB']:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'no specified recipe'):
                    encode_medium(name)
                self.assertEqual(len(encode_medium({'medium_id': name, 'composition': {'Glucose': 4}})['vector']), 159)

    def test_scientific_notation_and_rounding(self):
        self.assertAlmostEqual(self.sparse('MGm-2.5')['TSE'], 2.1e-5, places=11)
        self.assertAlmostEqual(self.sparse('MM-101')['TSE'], 9.2e-5, places=11)
        self.assertAlmostEqual(self.sparse('SSBM-P')['CoCl2·6H2O'], 1.1e-4, places=10)
        self.assertAlmostEqual(self.sparse('HdB-Tyl')['CaCl2'], .0068, places=8)

    def test_all_human_readable_recipes_encode(self):
        recipes = json.loads((ROOT/'configs/media_recipes.json').read_text())
        self.assertEqual(len(recipes), 40)
        self.assertEqual(len(self.catalog['media']), 38)
        for name, row in recipes.items():
            if row['composition'] is None:
                continue
            with self.subTest(name=name):
                named = encode_medium(name)['vector']
                explicit = encode_medium({'medium_id': name, 'composition': row['composition']})['vector']
                self.assertEqual(named, explicit)
                self.assertEqual(len(named), 159)
                self.assertEqual(np.count_nonzero(named), row['n_components'])
        example = json.loads((ROOT/'examples/medium_composition.json').read_text())
        self.assertEqual(encode_medium(example)['vector'], encode_medium('ISP2')['vector'])

    def test_invalid_inputs_fail(self):
        for spec in ['not_a_medium', {'medium_id': 'x', 'composition': {'unknown': 1}},
                     {'medium_id': 'x', 'composition': {'Casamino acids': -1}},
                     {'medium_id': 'x', 'vector': [0]*158}]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                encode_medium(spec)


if __name__ == '__main__':
    unittest.main()
