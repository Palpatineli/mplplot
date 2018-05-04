"""plotting infrastructure"""
from functools import partial

import numpy as np
from matplotlib import pyplot, rc
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from typing import List, Sequence, Tuple, Union, Dict

CLR_BACKGROUND = 'lightgrey'
CLR_FOREGROUND = 'white'
CLR_BLACK = 'black'
STAMP_LINE_COLOR = '#E6E6E6'
MARKER_COLOR = ['#C95278', '#913C94', '#C6DB59', '#8BCB52',
                '#6E0225', '#4E0251', '#657802', '#356F02']

DEFAULT_PARAMS = {
    'major_width': 1,
    'minor_width': 0.5,
    'marker_size': 3,
    'fig_size': 6.378,
    'marker': ['-'],
    'dpi': 300,
    'font': {'family': 'Arial', 'size': 12},
    'title_size': 22,
    'color': MARKER_COLOR
}


def _create_proxies(ax: Axes, legends: List[str]) -> Axes:
    marker_colors = iter(MARKER_COLOR)
    proxy_rect = partial(Rectangle, [0, 0], 1, 1)
    proxy = [[proxy_rect(fc=next(marker_colors)), item] for item in legends]
    ax.legend(*zip(*proxy), loc='upper right')
    return ax


class Plot(object):
    axes = None
    fig = None
    params = DEFAULT_PARAMS
    filePath = None
    x = None  # type: np.ndarray
    y = None  # type: np.ndarray

    def __init__(self, name: str=None):
        if name:
            self.params['name'] = name

    def __del__(self):
        if self.fig:
            pyplot.close(self.fig)

    def _setup(self):
        # noinspection PyArgumentList
        rc('font', **self.params['font'])
        rc('grid', color='#0000ff', linewidth=1, linestyle=':')
        if isinstance(self.params['fig_size'], Sequence):  # create canvas
            self.fig = pyplot.figure(figsize=self.params['fig_size'], dpi=self.params['dpi'])
        else:
            fig_size = (self.params['fig_size'], self.params['fig_size'])
            self.fig = pyplot.figure(figsize=fig_size, dpi=self.params['dpi'])

    def _draw(self):
        raise NotImplementedError

    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        """Update the data and parameters for the plot
        Args:
            data: data for plot, if multiple categories those categories are in rows of np.ndarray
            params: allowable fields ['x_label', 'y_label', 'legend', 'stamps', 'marker', 'title']
                'legend', 'stamps', 'marker' are List[str]
        """
        self.x, self.y = data
        self.params.update(params)

    def show(self):
        if not self.fig:
            self._setup()
        else:
            self.fig.clf()
        self._draw()
        pyplot.show()

    def save(self):
        if not self.filePath:
            raise IOError('file path not given!')
        if not self.fig:
            self._setup()
        else:
            self.fig.clf()
        self._draw()
        pyplot.savefig(self.filePath, format='svg')

    def close(self):
        pyplot.close(self.fig)

    def _ax(self, fig=None, location=None):
        if self.axes:
            return self.axes
        else:
            if not location:
                location = (1, 1, 1)
            if not fig:
                fig = self.fig
            axes = fig.add_subplot(*location)
            pyplot.cla()
            return axes

    def _add_label(self, ax: Axes, pos: Tuple[Tuple[float, float]]=(None, None, None)):
        labels = [ax.set_xlabel(self.params.get('x_label', '')),
                  ax.set_ylabel(self.params.get('y_label', '')),
                  ax.set_title(self.params.get('name', ''))]
        for idx, coord in enumerate(pos):
            if coord:
                labels[idx].set_position(coord)


class Histogram(Plot):
    """make a histogram given the binning points on x axis and the sample"""
    PADDING = 0.05
    major_width = 1.5

    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        super(Histogram, self).update(data, params)
        self.y = np.vstack([np.histogram(row, data[0])[0] for row in data[1]])

    def _draw(self):
        ax = self._ax()
        color_iter = iter(self.params['color'])
        step = (self.x[1] - self.x[0]) / self.y.shape[0]
        bins = self.x[0:-1]
        for row in self.y:
            bins += step / 2.0
            ax.bar(bins, row, color=next(color_iter), align='center', width=step, alpha=0.5)
        ax.set_xlim(self.x[0] * (1 + self.PADDING) - self.x[-1] * self.PADDING,
                    self.x[-1] * (1 + self.PADDING) - self.x[0] * self.PADDING)
        ax.set_ylim(0, self.y.max() * 1.2)

        if 'legend' in self.params:
            _create_proxies(ax, self.params['legend'])
        if 'stamps' in self.params:
            stamp_y, text_y = np.add(self.y.max(), [1.05, 1.15])
            for row in self.y:
                mean = np.multiply(row, bins + step / 2.0).sum() / row.sum()
                ax.annotate(str(mean), (mean, stamp_y), (mean, text_y),
                            arrowprops={'facecolor': 'black', 'shrink': 0.5})
        self._add_label(ax)


class BeforeAfter(Plot):
    """compare a data point under two circumstances (x and y axes) in a
    scatter plot"""
    EDGE = 0.2

    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        super(BeforeAfter, self).update(data, params)
        minimum = min((data[0].min(), data[1].min()))  # type: float
        maximum = max((data[0].max(), data[1].max()))  # type: float

        edge = (maximum - minimum) * self.EDGE
        self.params['range'] = (minimum - edge, maximum + edge)

    def _draw(self):
        ax = self._ax()
        data_range = self.params['range']  # type: Tuple[float, float]
        ax.scatter(self.x, self.y, cmap=self.params['marker'][0], alpha=0.5)
        ax.plot(data_range, data_range)
        ax.set_xlim(*data_range)
        ax.set_ylim(*data_range)
        self._add_label(ax)


class BoxPlot(Plot):  # TODO: deprecated
    WIDTH = 0.5
    FONT_SIZE = 14
    EDGE = 0.1

    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        self.params['marker'] = ['.']
        super(BoxPlot, self).update(data, params)
        raw_range = (min((row.min() for row in self.y)), max((row.max() for row in self.y)))
        edge_mat = [[1.0 + self.EDGE, -self.EDGE], [-self.EDGE, 1.0 + self.EDGE]]
        self.params['range'] = np.dot(edge_mat, raw_range)

    def _draw(self):
        ax = self._ax()
        from matplotlib.patches import Polygon
        ax.set_xlim(0.5, len(self.y) + 0.5)
        ax.set_ylim(self.params['range'][0], self.params['range'][1])

        bp = ax.boxplot(self.y, notch=True, sym=self.params['marker'][0], vert=True)
        pyplot.setp(list(ax.spines.values()), color=CLR_BACKGROUND)
        pyplot.setp(bp['fliers'], color=CLR_BLACK)
        ax.set_facecolor(CLR_BACKGROUND)
        ax.yaxis.grid(True, linestyle='-', which='major', color=CLR_FOREGROUND, alpha=0.5)
        ax.set_axisbelow(True)
        color = self.params['color']
        for idx in range(len(self.y)):
            box = bp['boxes'][idx]
            box_coords = [(box.get_xdata()[i], box.get_ydata()[i])
                          for i in range(11)]
            box_polygon = Polygon(box_coords, facecolor=color[idx])
            ax.add_patch(box_polygon)
            pyplot.setp(box, color=color[idx])
            median = bp['medians'][idx]
            pyplot.setp(median, color=CLR_FOREGROUND)
            for j in range(2):
                whisker = bp['whiskers'][idx * 2 + j]
                pyplot.setp(whisker, color=color[idx], linewidth=3, linestyle='-')
                cap = bp['caps'][idx * 2 + j]
                pyplot.setp(cap, color=color[idx], linewidth=3, linestyle='-')
        x_label = pyplot.setp(ax, xticklabels=self.x)
        pyplot.setp(x_label, rotation=13, fontsize=self.params['font']['size'])
        return ax


class LinePlot(Plot):
    """plain simple line plots that allows multiple lines up to 5. Also shaded
    or just marked x axis regions. similar to sns.tsplot
    """
    def update(self, data: Tuple[np.ndarray, np.ndarray], params: Dict[str, Union[str, List[str]]]):
        super(LinePlot, self).update(data, params)
        self.params.setdefault('legend', [''] * self.y.shape[0])

    def _draw(self):
        ax = self._ax()
        ax.set_xlim(min(self.x), max(self.x))
        colors = iter(self.params['color'])
        legend = iter(self.params['legend'])

        for idx, row in enumerate(self.y[0:4, :]):
            ax.plot(self.x, row, color=next(colors), linestyle='-', label=next(legend))
        ax.legend(prop={'size': 12})
        if 'stamp' in self.params and isinstance(self.params['stamp'], Sequence):
            # noinspection PyTypeChecker
            for stamp in self.params['stamp']:
                if isinstance(stamp, Sequence):
                    ax.axvspan(stamp[0], stamp[1], color=STAMP_LINE_COLOR)
                else:
                    ax.axvline(stamp, color=STAMP_LINE_COLOR)
        self._add_label(ax)
        return ax
