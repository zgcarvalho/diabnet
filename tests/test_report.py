import numpy as np
import notebooks.report_ensemble as report

def test_ece():
    preds = np.arange(0.0,1.0,.01)
    labels = np.arange(0.0,1.0,.01)
    ece = report.ece(labels, preds)
    assert ece == 0.0

def test_mce():
    preds = np.arange(0.0,1.0,.01)
    labels = np.arange(0.0,1.0,.01)
    mce = report.mce(labels, preds)
    assert mce == 0.0


