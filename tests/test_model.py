import unittest
import torch
from diabnet.model import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.baseline = 0.121
        self.topline = 0.8
        self.slope = -0.12/120
        self.model = Model(100, 3, 0.5, "lc1", True, self.baseline, self.topline, self.slope, 0)
        self.x = torch.rand((1000, 2, 100))

    def test_forward(self):
        y = self.model(self.x)
        print(y)
        self.assertTrue(True)

    def test_apply(self):
        y = self.model.apply(self.x)
        print(y)
        self.assertTrue(True)

    def test_sigmoid_without_correction(self):
        y = torch.unsqueeze(torch.arange(-7.0,7.0,1.0),1)
        ages = torch.ones(len(y))
        sig = self.model.sigmoid(y, ages, with_correction=False)
        print(sig)
        self.assertTrue(True)

    def test_sigmoid_with_correction_slope_zero_topline(self):
        slope = self.model.soft_label_baseline_slope
        self.model.soft_label_baseline_slope = 0.0
        y_ones = torch.logit(torch.ones(5,1) * self.topline)
        ages = torch.ones(len(y_ones))
        sig = self.model.sigmoid(y_ones, ages, with_correction=True)
        print(sig)
        # restore slope
        self.model.soft_label_baseline_slope = slope
        self.assertTrue((sig == 1.0).all().item())

    def test_sigmoid_with_correction_slope_zero_baseline(self):
        slope = self.model.soft_label_baseline_slope
        self.model.soft_label_baseline_slope = 0.0
        y_zeros = torch.logit(torch.ones(5,1) * self.baseline)
        ages = torch.ones(len(y_zeros))
        sig = self.model.sigmoid(y_zeros, ages, with_correction=True)
        print(sig)
        # restore slope
        self.model.soft_label_baseline_slope = slope
        self.assertTrue((sig == 0.0).all().item())

    def test_sigmoid_with_correction_slope_zero_middle(self):
        slope = self.model.soft_label_baseline_slope
        self.model.soft_label_baseline_slope = 0.0
        y_half = torch.logit(torch.ones(5,1) * ((self.topline - self.baseline)/2 + self.baseline)) 
        ages = torch.ones(len(y_half))
        sig = self.model.sigmoid(y_half, ages, with_correction=True)
        print(sig)
        # restore slope
        self.model.soft_label_baseline_slope = slope
        self.assertTrue((sig == 0.5).all().item())

    def test_sigmoid_topline_slope(self):
        y = torch.logit(torch.ones(10,1) * self.topline)
        ages = torch.unsqueeze(torch.arange(.2, 2.2, .2), 1)
        sig = self.model.sigmoid(y, ages, with_correction=True)
        print(sig)
        self.assertTrue((sig == 1.0).all().item())

    def test_sigmoid_baseline_slope(self):
        y = torch.logit(torch.ones(10,1) * self.baseline)
        ages = torch.unsqueeze(torch.arange(.0, 2., .2), 1)
        sig = self.model.sigmoid(y, ages, with_correction=True)
        print(sig)
        self.assertTrue(torch.all(torch.diff(sig) < 0.0).item())