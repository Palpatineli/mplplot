from typing import Tuple
import numpy as np
from matplotlib.pyplot import Axes

def bootstrap(data: np.ndarray, repeat: int = 1000, ci: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Given a 2-D matrix with rows as observation and columns as features,
    returns lower and upper obund 1-D matrixs of all features. Here ci is the sum of both tails.
    """
    shape_0, shape_1 = data.shape
    idx = np.random.randint(0, shape_0, (repeat, shape_0, shape_1))
    idx2 = np.tile(np.arange(shape_1), (repeat, shape_0, 1))
    upper_idx, lower_idx = int(round((1.0 - ci * 0.5) * repeat)), int(round(ci * 0.5 * repeat))
    result = np.sort(data[idx, idx2].mean(1), 0)
    return result[lower_idx, :], result[upper_idx, :]

def main(ax: Axes, data: np.ndarray, time: np.ndarray = None, **kwargs) -> Axes:
    if time is None or time.ndim != 1 or time.size < data.shape[1]:
        time = np.arange(data.shape[1])
    elif time.size > data.shape[1]:
        time = time[0: data.shape[1]]
    ave = data.mean(0)
    lower, upper = bootstrap(data)
    ax.fill_between(time, lower, upper, alpha=0.2, **kwargs)
    ax.plot(time, ave, **kwargs)
    ax.margins(x=0)
    return ax
