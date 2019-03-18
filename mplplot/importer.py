from typing import Tuple, Dict, Optional
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class MplFigure(object):
    ax: Optional[plt.Axes] = None

    def __init__(self, save_path: Optional[str] = None, figsize: Tuple[float, float] = (6, 6),
                 despine: Optional[Dict[str, bool]] = None, **kwargs) -> None:
        sns.set(context="paper", style="ticks", palette="husl", font="Arial", font_scale=3)
        plt.rcParams['svg.fonttype'] = 'none'  # do not convert font to path in svg output
        plt.ioff()
        self.figsize = figsize
        self.save_path = save_path
        self.is_despine = {'top': True, 'bottom': False, 'left': False, 'right': True,
                           **(despine if despine else dict())}
        self.kwargs = kwargs

    def despine(self) -> None:
        edges = self.is_despine
        if self.ax is None:
            raise ValueError("despining unintialized ax!")
        if edges:
            sns.despine(**edges)
            for edge, func in (('bottom', 'get_xticklines'), ('left', 'get_yticklines')):
                if edges.get(edge):
                    if isinstance(self.ax, np.ndarray):
                        for single_ax in self.ax:
                            for tick in getattr(single_ax, func)():
                                tick.set_visible(False)
                    else:
                        for tick in getattr(self.ax, func)():
                            tick.set_visible(False)
        else:
            sns.despine()

    def __enter__(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=100, **self.kwargs)
        self.fig = fig
        self.ax = ax
        return ax

    def __exit__(self, type, value, traceback):
        self.despine()
        if self.save_path is not None:
            self.fig.savefig(self.save_path)
            plt.close("all")
        else:
            plt.show()
