import os
import pickle
import unittest
import random
import diabnet.data as data

TEST_FILE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestData(unittest.TestCase):
    def test_get_feature_names(self):
        # Get example data
        fn = os.path.join(TEST_FILE_DIR, "example_data.csv")
        # Default arguments
        result = data.get_feature_names(fn)
        with open(os.path.join(TEST_FILE_DIR, "feature_names.data"), "rb") as file:
            expected = pickle.load(file)
        self.assertListEqual(result, expected)
        # use_sex = False
        result = data.get_feature_names(fn, use_sex=False)
        expected.remove("sex")
        self.assertListEqual(result, expected)
        # use_parents_diagnosis = False
        result = data.get_feature_names(fn, use_parents_diagnosis=False)
        with open(os.path.join(TEST_FILE_DIR, "feature_names.data"), "rb") as file:
            expected = pickle.load(file)
        for feature in ["mo_t2d", "fa_t2d"]:
            expected.remove(feature)
        self.assertListEqual(result, expected)

    def test_encode_features(self):
        # Test snp
        # Dominant homozygous
        result = data.encode_features(["snp"], [0]).tolist()
        self.assertListEqual(result, [[2.0], [0.0]])
        # Heterozygous
        result = data.encode_features(["snp"], [1]).tolist()
        self.assertListEqual(result, [[1.0], [1.0]])
        # Recessive homozygous
        result = data.encode_features(["snp"], [2]).tolist()
        self.assertListEqual(result, [[0.0], [2.0]])
        # Test AGE
        result = data.encode_features(["AGE"], [20]).tolist()
        self.assertListEqual(result, [[0.4], [1.0]])
        result = data.encode_features(["AGE"], [40]).tolist()
        self.assertListEqual(result, [[0.8], [1.0]])
        result = data.encode_features(["AGE"], [60]).tolist()
        self.assertListEqual(result, [[1.2], [1.0]])
        result = data.encode_features(["AGE"], [80]).tolist()
        self.assertListEqual(result, [[1.6], [1.0]])
        # Test sex
        result = data.encode_features(["sex"], ["F"]).tolist()
        self.assertListEqual(result, [[2.0], [0.0]])
        result = data.encode_features(["sex"], ["M"]).tolist()
        self.assertListEqual(result, [[0.0], [2.0]])
        # Test mo_t2d and fa_t2d
        result = data.encode_features(["snp", "mo_t2d", "fa_t2d"], [0, 0, 0]).tolist()
        self.assertListEqual(
            result,
            [[2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        result = data.encode_features(["snp", "mo_t2d", "fa_t2d"], [0, 0, 1]).tolist()
        self.assertListEqual(
            result,
            [[2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        result = data.encode_features(["snp", "mo_t2d", "fa_t2d"], [0, 0, 2]).tolist()
        self.assertListEqual(
            result,
            [[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        result = data.encode_features(["snp", "mo_t2d", "fa_t2d"], [0, 1, 0]).tolist()
        self.assertListEqual(
            result,
            [[2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        result = data.encode_features(["snp", "mo_t2d", "fa_t2d"], [0, 2, 0]).tolist()
        self.assertListEqual(
            result,
            [[2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        )
        # Test unknown feature
        self.assertRaises(ValueError, data.encode_features, ["unknown-feature"], [0.0])

    def test_check_parents_diag(self):
        # Wrong order for 'mo_t2d' and 'fa_t2d'
        self.assertRaises(
            ValueError, data._check_parents_diag, ["snp", "fa_t2d", "mo_t2d"]
        )
        self.assertRaises(
            ValueError, data._check_parents_diag, ["mo_t2d", "fa_t2d", "snp"]
        )
        self.assertRaises(
            ValueError, data._check_parents_diag, ["fa_t2d", "mo_t2d", "snp"]
        )
        # Correct order
        self.assertIsNone(data._check_parents_diag(["mo_t2d", "fa_t2d"]))


if __name__ == "__main__":
    unittest.main()
