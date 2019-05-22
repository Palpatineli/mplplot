from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def _scale_f(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    return (x - x_min) * (1 / (x_max - x_min))

def _scale_b(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    x_new = ((x - x_min) * (256 / (x_max - x_min)))
    x_new[x_new >= 256] = 255
    return x_new.astype(np.uint8)

def _segmentcolor(data: np.ndarray, colors: Dict[str, List[Tuple[float, float, float]]]) -> np.ndarray:
    keys = np.array([x[0] for x in colors['red']])
    values = np.array([[x[1] for x in colors[key]] for key in ('red', 'green', 'blue', 'alpha')]).T
    idx = np.searchsorted(keys, data)
    pre_key, post_key = keys[idx - 1], keys[idx]
    res = ((data - pre_key)[:, :, :, np.newaxis] * values[idx]
           + (post_key - data)[:, :, :, np.newaxis] * values[idx - 1]) / (post_key - pre_key)[:, :, :, np.newaxis]
    res[data == 1.0] == values[-1]
    return res

def voxel(ax: Axes, data: np.ndarray, cmap: str = 'viridis', **kwargs) -> Axes:
    cmap_obj = plt.get_cmap(cmap)
    if hasattr(cmap_obj, "_segmentdata"):
        facecolors = _segmentcolor(_scale_f(data), cmap_obj._segmentdata)
    elif hasattr(cmap_obj, "colors"):
        facecolors = np.asarray(cmap_obj.colors)[_scale_b(data), :]
    ax.voxels(np.ones(data.shape, dtype=np.bool_), facecolors=facecolors, **kwargs)
    return ax
