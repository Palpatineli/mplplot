from typing import Tuple, Dict, Optional, List, Sequence
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

Sequence.register(np.ndarray)

def _is_list(x):
    return not isinstance(x, str) and isinstance(x, Sequence)

class Figure(object):
    def __init__(self, save_path: Optional[str] = None, figsize: Tuple[float, float] = (6, 6),
                 grid: Tuple[int, int] = (1, 1), projection=None,
                 despine: Optional[Dict[str, bool]] = None,
                 font_scale: int = 3,
                 show: bool = False,
                 **kwargs) -> None:
        sns.set(context="paper", style="ticks", palette="husl", font="Arial", font_scale=font_scale)
        plt.rcParams['svg.fonttype'] = 'none'  # do not convert font to path in svg output
        self.figsize = figsize
        self.save_path = save_path
        self.is_despine = {'top': True, 'bottom': False, 'left': False, 'right': True,
                           **(despine if despine else dict())}
        self.kwargs = kwargs
        self.grid = grid
        self.show = show
        self.proj = projection if _is_list(projection) else ([projection] * (grid[0] * grid[1]))

    def despine(self) -> None:
        edges = self.is_despine
        if not self.axes:
            raise ValueError("despining empty axes!")
        if edges:
            sns.despine(**edges)
            for edge, func in (('bottom', 'get_xticklines'), ('left', 'get_yticklines')):
                if edges.get(edge):
                    for ax in self.axes:
                        for tick in getattr(ax, func)():
                            tick.set_visible(False)
        else:
            sns.despine()

    def __enter__(self):
        plt.ioff()
        plt.close('all')
        fig = plt.figure(figsize=self.figsize, dpi=100, **self.kwargs)
        self.axes: List[plt.Axes] = list()
        size_y, size_x = self.grid
        for grid_y in range(size_y):
            for grid_x in range(size_x):
                idx = grid_y * size_x + grid_x
                ax = fig.add_subplot(size_y, size_x, idx + 1, projection=self.proj[idx])
                self.axes.append(ax)
        self.fig = fig
        return self.axes

    def __exit__(self, type, value, traceback):
        self.despine()
        if self.save_path is None or self.show:
            plt.show()
        if self.save_path is not None:
            self.fig.savefig(self.save_path)
        plt.ion()
