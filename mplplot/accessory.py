from typing import Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.container import Container

def add_scalebar(ax: Axes, xy: Tuple[float, float, float, float], linewidth: float) -> Container:
    """
    Args:
        xy: [x0, y0, x1, y1]
    """
    center = np.add(xy[: 2], xy[2:]) / 2
    width, height = np.subtract(xy[2:], xy[: 2])
    half_width, half_height = np.subtract(xy[2:], xy[: 2]) / 2
    rects = Container([
        Rectangle(xy[0: 2], half_width, half_height, fill=True, linewidth=None, facecolor="#FFFFFF"),
        Rectangle((xy[0], center[1]), half_width, half_height, fill=True, linewidth=None, facecolor="#000000"),
        Rectangle(center, half_width, half_height, fill=True, linewidth=None, facecolor="#FFFFFF"),
        Rectangle((center[0], xy[1]), half_width, half_height, fill=True, linewidth=None, facecolor="#000000"),
        Rectangle(xy[0: 2], width, height, fill=False, linewidth=linewidth, edgecolor="#000000")
    ])
    ax.add_container(rects)
    [ax.add_patch(rect) for rect in rects]
    return rects

