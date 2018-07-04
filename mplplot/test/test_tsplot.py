from pytest import fixture
import numpy as np
import matplotlib.pyplot as plt
from mplplot.tsplot import main as tsplot

@fixture
def sine_wave():
    """10x31 matrix of 10 sine waves with trial and sampling errors"""
    old_state = np.random.get_state()
    np.random.seed(12345)
    x = np.linspace(0, 15, 31)
    data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    yield data
    np.random.set_state(old_state)

def test_tsplot(sine_wave):
    _, ax = plt.subplots()
    ax = tsplot(ax, sine_wave)
