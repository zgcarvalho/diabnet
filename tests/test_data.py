import numpy as np
from diabnet.data import DiabDataset

def test_soft_label():
    labels = np.array([0,0,0,0,0,1,1,1,1,1])
    ages = np.arange(10,101,10)
    soft_label_ajusted_per_age = DiabDataset._soft_label(labels, ages, 0.4, 0.99,-0.6)
    assert np.all(np.less(soft_label_ajusted_per_age[:5], 0.5))
    assert np.all(np.greater_equal(soft_label_ajusted_per_age[:5], 0.0))
    assert np.all(np.greater(soft_label_ajusted_per_age[5:], 0.5))
    assert np.all(np.greater(soft_label_ajusted_per_age[5:], 0.98))
    assert np.all(np.less_equal(soft_label_ajusted_per_age[5:], 1.0))
    assert soft_label_ajusted_per_age[0] > soft_label_ajusted_per_age[4]
    
def test_soft_label_without_slope():
    labels = np.array([0,0,0,0,0,1,1,1,1,1])
    ages = np.arange(10,101,10)
    soft_label_ajusted_per_age = DiabDataset._soft_label(labels, ages, 0.4, 0.99, 0.0)
    assert np.all(np.less(soft_label_ajusted_per_age[:5], 0.5))
    assert np.all(np.greater_equal(soft_label_ajusted_per_age[:5], 0.0))
    assert np.all(np.greater(soft_label_ajusted_per_age[5:], 0.5))
    assert np.all(np.greater(soft_label_ajusted_per_age[5:], 0.98))
    assert np.all(np.less_equal(soft_label_ajusted_per_age[5:], 1.0))
    assert soft_label_ajusted_per_age[0] == soft_label_ajusted_per_age[4]
 
