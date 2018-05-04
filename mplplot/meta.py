import numpy as np
from matplotlib import pyplot
from typing import Tuple, List

from plptn.plot import Plot


CHAR_A = ord('A')


class MetaPlot(Plot):
    panel_list = list()
    shape = (1, 1)

    def __del__(self):
        del(self.panel_list[:])

    @staticmethod
    def _get_factor(x: int) -> Tuple[int, int]:
        max_factor = 1
        for a in range(int(np.sqrt(x)), 1, -1):
            if x % a == 0:
                max_factor = a
        return max_factor, x // max_factor

    def update(self, panel_list: List[Plot], shape: Tuple[int, int]=None):
        self.panel_list = panel_list
        self.shape = shape if shape else self._get_factor(len(panel_list))

    def show(self):
        self._setup()
        self._draw()
        pyplot.show()

    def save(self):
        self._setup()
        self._draw()
        pyplot.savefig(self.filePath, format='svg')

    def _draw(self):
        shape = self.shape
        for idx, panel in enumerate(self.panel_list):
            subplot = self.panel_list[idx]
            if not subplot:
                continue
            location = (idx // shape[1], idx % shape[1], idx + 1)
            subplot.params['name'] = None
            subplot.axes = subplot._ax(self.fig, location)
            subplot._draw()
            subplot.axes.set_title(
                chr(idx + CHAR_A), {'fontsize': self.params['title_size']}, 'left')
        self.fig.tight_layout()
