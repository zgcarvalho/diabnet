import unittest
import numpy as np
import torch
from diabnet.metrics import ece_mce


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.targets = torch.arange(0.0, 1.0, 0.01)
        self.predictions = torch.arange(0.0, 1.0, 0.01)

    def test_mce_ece(self):
        # torch.Tensor inputs
        result = ece_mce(self.predictions, self.targets, 10)
        self.assertTupleEqual(result, (0.0, 0.0))

    def test_bad_types(self):
        # numpy.ndarray inputs
        self.assertRaises(
            TypeError, ece_mce, np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01)
        )
        # List inputs
        self.assertRaises(
            TypeError,
            ece_mce,
            np.arange(0.0, 1.0, 0.01).tolist(),
            np.arange(0.0, 1.0, 0.01).tolist(),
        )
