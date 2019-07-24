from typing import Optional, List
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot
from mplplot.importer import Figure

HISTOGRAM_RATIO = 0.2  # length vs. height ratio of the histogram

def pair(xs: np.ndarray, ys: np.ndarray, colors: List[str], ax: Optional[Axes] = None, **kwargs) -> Axes:
    """Draw a scatterplot for two features of same observations.
    Args:
        xs, ys: list of 1d arrays,
            each array is one groups, each array iteim is one observations.
            xs and ys are two features of the same observations, so must have same shape.
            between groups they don't need same shape
        colors: a list of colors for the different groups.
        ax: the axes to draw
    Returns:
        The main axes with the scatterplot. Now is has .sup_ax which the aux_ax of the histgram.
    """
    if ax is None:
        ax = plt.gca()
    # get density, and extremies of both density and scatter
    xyds = [density_plot(x - y, **kwargs) for x, y in zip(xs, ys)]
    yd_max = max([y.max() for _, y in xyds])
    xd_max = max(abs(min([x.min() for x, _ in xyds])), abs(max([x.max() for x, _ in xyds])))
    x_minmax = (min(min([x.min() for x in xs]), min([y.min() for y in ys])),
                max(max([x.max() for x in xs]), max([y.max() for y in ys])))
    x_range = x_minmax[1] - x_minmax[0]
    # calculate size of scatter plot and desnity plot
    r_b = 0.8 / (1 + (0.5 + HISTOGRAM_RATIO) * (xd_max / x_range))
    r_l = (0.8 - r_b) * (1 + 1 / (1 + 2 * HISTOGRAM_RATIO))
    size = -xd_max, xd_max, 0, yd_max  # change size
    # generate density plot axes, set size for both plots
    tr = Affine2D().scale(0.5 / xd_max, HISTOGRAM_RATIO / yd_max).rotate_deg(-45)
    sup_ax = FloatingSubplot(ax.figure, 111, grid_helper=GridHelperCurveLinear(tr, size))
    sup_ax_aux = sup_ax.get_aux_axes(tr)
    ax.set_position([0.1, 0.1, r_b, r_b])
    sup_ax.set_position([0.9 - r_l, 0.9 - r_l, r_l, r_l])
    [x.set_visible(False) for x in sup_ax.spines.values()]
    [x.set_visible(False) for x in sup_ax_aux.spines.values()]
    # draw density
    for (x, y), color in zip(xyds, colors):
        sup_ax_aux.fill(x, y, color=color, alpha=0.75)
    ax.figure.add_subplot(sup_ax)
    ax.sup_ax = sup_ax_aux
    # draw scatterplot
    for x0, y0, color in zip(xs, ys, colors):
        ax.scatter(x0, y0, color=color)
    ax.set_xlim(*x_minmax)
    ax.set_ylim(*x_minmax)
    ax.plot(x_minmax, x_minmax, ls='--', c='.3')
    return ax

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
