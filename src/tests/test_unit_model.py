# -*- coding: utf-8 -*-
import unittest
import src.models.deep_api as deep_api

class TestModelMethods(unittest.TestCase):
    """Default tests for Deep Hybrid DataCloud."""

    def setUp(self):
        self.meta = deep_api.get_metadata()

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['name'].lower().replace('-','_'),
                         'conex-generator'.lower().replace('-','_'))
        self.assertEqual(self.meta['author'].lower(),
                         'Marcel Köpke, KIT/IKP'.lower())
        self.assertEqual(self.meta['author-email'].lower(),
                         'marcel.koepke@kit.edu'.lower())
        #self.assertEqual(self.meta['license'].lower(),
        #                 'BSD-3-Clause'.lower())


if __name__ == '__main__':
    unittest.main()

