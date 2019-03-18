from typing import Tuple, Union, List, Any, Sequence
import numpy as np
from matplotlib.pyplot import Axes
from matplotlib.container import BarContainer

def _bootstrap(data: np.ndarray, repeat: int = 1000, ci: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Given a 2-D matrix with rows as observation and columns as features,
    returns lower and upper obund 1-D matrixs of all features. Here ci is the sum of both tails.
    """
    shape_0, shape_1 = data.shape
    idx = np.random.randint(0, shape_0, (repeat, shape_0, shape_1))
    idx2 = np.tile(np.arange(shape_1), (repeat, shape_0, 1))
    upper_idx, lower_idx = int(round((1.0 - ci * 0.5) * repeat)), int(round(ci * 0.5 * repeat))
    result = np.sort(data[idx, idx2].mean(1), 0)
    return result[lower_idx, :], result[upper_idx, :]

def tsplot(ax: Axes, data: np.ndarray, time: np.ndarray = None, **kwargs) -> Axes:
    if time is None or time.ndim != 1 or time.size < data.shape[1]:
        time = np.arange(data.shape[1])
    elif time.size > data.shape[1]:
        time = time[0: data.shape[1]]
    ave = data.mean(0)
    if 'ci' in kwargs:
        lower, upper = _bootstrap(data, ci=kwargs.pop('ci'))
    else:
        lower, upper = _bootstrap(data)
    ax.fill_between(time, lower, upper, alpha=0.2, **kwargs)
    ax.plot(time, ave, **kwargs)
    ax.margins(x=0)
    return ax

def _fill_zeros(data: List[List[Any]]) -> np.ndarray:
    """Create ndarray of [len(data), max(len(data[x]))], and fill the nonexisting data as zero."""
    length = max(len(x) for x in data)
    result = np.zeros((len(data), length), dtype=type(data[0][0]))
    for row, sublist in zip(result, data):
        row[:len(sublist)] = sublist
    return result

def stacked_bar(ax: Axes, data: Union[np.ndarray, List[List[Any]]], colors: Sequence[str],
                **kwargs) -> List[BarContainer]:
    """Draw stacked bar graph in ax.
    Args:
        ax: the axes to draw in
        data: either a matrix with rows of series/categories or a list of series
        colors: series of colors for the series of data
    """
    if not isinstance(data, np.ndarray):
        data = _fill_zeros(data)
    x_axis = np.arange(data.shape[0])
    width = kwargs.pop('width', 0.35)
    ax.bar(x_axis, data[:, 0], width, color=colors[0], **kwargs)
    bars = list()
    for row, bottom, color in zip(data.T[1:], np.cumsum(data[0: -1]), colors[1:]):
        bars.append(ax.bar(x_axis, row, width, bottom=bottom, color=color))
    return bars
