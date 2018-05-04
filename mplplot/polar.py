import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from typing import Tuple, Dict, Union, List

from main import _create_proxies
from plptn.plot import Plot


class Polar(Plot):
    """Draw a plot on polar coordinates, usually a perimeter chart"""
    def _draw(self):
        raise NotImplementedError

    def _ax(self, fig=None, location=None):
        """provide the class with an axes in polar coordinates"""
        if self.axes:
            return self.axes
        if not location:
            location = (1, 1, 1)
        if not fig:
            fig = self.fig
        axes = fig.add_subplot(*location, projection='polar')
        pyplot.cla()
        return axes

    def _add_label(self, ax: Axes, pos: Tuple[float, float] = None):
        title = ax.set_title(self.params.get('name', ''))
        if pos:
            title.set_position(pos)


class Radar(Polar):
    """polar plot of multiple lines"""
    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        """data
        Args:
            data: [x, y], x: a list of angles. y: a 2-D array with each row being one condition
            params: regular params dict, see Plot
        """
        super(Radar, self).update(data, params)
        group_no = data[1].shape[0]
        self.params['marker'] *= group_no
        if 'marker' in params:
            common_length = min(len(params['marker']), len(group_no))
            self.params['marker'][0:common_length] = params['marker'][0: common_length]

    def _draw(self):
        ax = self._ax()
        if self.x.ndim() < 2 or self.x.shape[0] == 1:
            self.x = np.tile(self.x, (self.y.shape[0], 1))
        for idx, row in enumerate(self.y):
            x = np.append(self.x[idx], self.x[idx][0])
            y = np.append(row, row[0])
            ax.plot(x, y, self.params['marker'][idx], color=self.params['color'][idx],
                    linewidth=self.params['major_width'], markersize=self.params['marker_size'])
        if 'legend' in self.params:
            ax = _create_proxies(ax, self.params['legend'])
        self._add_label(ax)
