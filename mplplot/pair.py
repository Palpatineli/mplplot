from typing import Optional, List
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot
from mplplot.importer import Figure

def pair(xs: np.ndarray, ys: np.ndarray, colors: List[str], ax: Optional[Axes] = None) -> Axes:
    xs, ys = np.asarray(xs).squeeze(), np.asarray(ys).squeeze()
    if ax is None:
        ax = plt.gca()
    xyds = [density_plot(x - y) for x, y in zip(xs, ys)]
    yd_max = max([y.max() for _, y in xyds])
    xd_max = max(abs(min([x.min() for x, _ in xyds])), abs(max([x.max() for x, _ in xyds])))
    x_minmax = min(xs.min(), ys.min()), max(xs.max(), ys.max())
    x_range = x_minmax[1] - x_minmax[0]
    r_b = 0.8 / (1 + 0.7 * (xd_max / x_range))
    r_l = (0.8 - r_b) * (12 / 7)
    size = -xd_max, xd_max, 0, yd_max  # change size
    tr = Affine2D().scale(0.5 / xd_max, 0.2 / yd_max).rotate_deg(-45)
    sup_ax = FloatingSubplot(ax.figure, 111, grid_helper=GridHelperCurveLinear(tr, size))
    sup_ax_aux = sup_ax.get_aux_axes(tr)
    ax.set_position([0.1, 0.1, r_b, r_b])
    sup_ax.set_position([0.9 - r_l, 0.9 - r_l, r_l, r_l])
    [x.set_visible(False) for x in sup_ax.spines.values()]
    [x.set_visible(False) for x in sup_ax_aux.spines.values()]
    for (x, y), color in zip(xyds, colors):
        sup_ax_aux.fill(x, y, color=color, alpha=0.75)
    ax.figure.add_subplot(sup_ax)
    ax.sup_ax = sup_ax
    for x0, y0, color in ([xs, ys, colors[0]]) if xs.ndim == 1 or ys.ndim == 1 else (zip(xs, ys, colors)):
        ax.scatter(x0, y0, color=color)
    ax.set_xlim(*x_minmax)
    ax.set_ylim(*x_minmax)
    ax.plot(x_minmax, x_minmax, ls='--', c='.3')

def density_plot(x, edge: float = 0.25, bw: float = 0.15, sample_no: int = 500):
    x_min, x_max = x.min(), x.max()
    xlim = (x_min - (x_max - x_min) * edge, x_max + (x_max - x_min) * edge)
    density_fn = gaussian_kde(x)
    density_fn.set_bandwidth(bw)
    x0 = np.linspace(*xlim, sample_no)
    density = density_fn(x0)
    return x0, density

def test_density_plot():
    np.random.randn(12345)
    x = [np.random.uniform(0.1, 0.9, 100), np.random.uniform(0.1, 0.9, 100)]
    y = [x[0] + np.random.randn(100) * 0.05, x[1] * 0.8 - 0.05 + np.random.randn(100) * 0.05]
    with Figure(figsize=(12, 12)) as axes:
        pair(x, y, ["#619CFF", "#00BA38"], axes[0])
