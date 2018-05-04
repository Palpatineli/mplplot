"""plot stacked bars using matplotlib, from numpy array"""
from typing import Sequence, Union, List
from collections import deque  # to consume Iterable with side effect
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def demo():
    d = np.array([[101., 0., 0., 0., 0., 0., 0.],
                  [92., 3., 0., 4., 5., 6., 0.],
                  [56., 7., 8., 9., 23., 4., 5.],
                  [81., 2., 4., 5., 32., 33., 4.],
                  [0., 45., 2., 3., 45., 67., 8.],
                  [99., 5., 0., 0., 0., 43., 56.]])

    d_heights = [1., 2., 3., 4., 5., 6.]
    d_widths = [.5, 1., 3., 2., 1., 2.]
    d_labels = ["fred", "julie", "sam", "peter", "rob", "baz"]
    d_colors = ['#2166ac', '#fee090', '#fdbb84', '#fc8d59', '#e34a33', '#b30000', '#777777']
    gap = 0.05

    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    plot(ax1,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels)
    plt.title("Straight up stacked bars")

    ax2 = fig.add_subplot(322)
    plot(ax2,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels,
         scale=True)
    plt.title("Scaled bars")

    ax3 = fig.add_subplot(323)
    plot(ax3,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels,
         heights=d_heights,
         y_ticks=7)
    plt.title("Bars with set heights")

    ax4 = fig.add_subplot(324)
    plot(ax4,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels,
         y_ticks=7,
         widths=d_widths,
         scale=True)
    plt.title("Scaled bars with set widths")

    ax5 = fig.add_subplot(325)
    plot(ax5,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels,
         gap=gap)
    plt.title("Straight up stacked bars + gaps")

    ax6 = fig.add_subplot(326)
    plot(ax6,
         d,
         d_colors,
         edge_colors=['#000000'] * 7,
         x_labels=d_labels,
         scale=True,
         gap=gap,
         end_gaps=True)
    plt.title("Scaled bars + gaps + end gaps")

    # We change the fontsize of minor ticks label
    fig.subplots_adjust(bottom=0.4)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    del fig


def plot(ax: Axes, data: np.ndarray, colors: Sequence[str],
         x_labels: Union[None, Sequence[str]]=None, y_ticks: Union[int, List[float]]=6.,
         edge_colors: List[str]=None, scale: bool=False, widths: List[float]=None,
         heights: List[float]=None, y_label: str=None,
         x_label: str=None, gap: float=0., end_gaps: bool=False) -> None:
    """plot function
    Args:
        ax: axes in matplotlib to plot on
        data: 2d array of data, each row is one series/category
        colors: list of colors to use for different categories
        x_labels: list of labels for x ticks
        y_ticks: y ticks
            information used for making y ticks ["none", <int> or [[tick_pos1, tick_pos2, ... ],
                                                 [tick_label_1, tick_label2, ...]]
        edge_colors: colors for edges
        scale: scale bars to same height
        widths: set widths for each bar
        heights: set heights for each bar
        y_label: label for x axis
        x_label: label for y axis
        gap: gap between bars
        end_gaps: allow gaps at end of bar chart (only used if gaps != 0.)
    """
    data_copy = data.copy().transpose()
    if widths is None:
        widths = np.full(data_copy.shape[1], 1)
        x = np.arange(data_copy.shape[1])
    else:
        widths = np.array(widths)
        x = np.concatenate([[0], np.cumsum(np.divide(widths[1:] + widths[0: -1], 2))])

    data_stack = np.cumsum(data_copy, axis=0, dtype=np.float)  # stack the data
    summation = data_stack[-1, :]
    data_stack = np.vstack([np.full(data_stack.shape[1], 0), data_stack[0: -1]])

    if scale or heights:  # scale the data if needed
        scale_factor = np.divide(heights if heights else 1.0, summation)
        data_copy *= scale_factor
        data_stack *= scale_factor

    if y_ticks:
        if not isinstance(y_ticks, Sequence):
            y_max = 1 if scale else np.max(heights) if heights else summation.max()
            ticks_at = np.linspace(0, y_max, int(y_ticks), endpoint=True)
            tick_labels = np.array(["{:.2f}".format(i) for i in ticks_at * (100 if scale else 1)])
            y_ticks = (ticks_at, tick_labels)
    else:
        y_ticks = ([], [])

    if edge_colors is None:
        edge_colors = ["none"] * len(colors)

    color = iter(colors)
    edge_color = iter(edge_colors)
    for top, bottom in zip(data_copy, data_stack):
        ax.bar(x, top, bottom=bottom, color=next(color), edgecolor=next(edge_color),
               width=np.subtract(widths, gap), linewidth=0.5, align='center')

    deque((x.set_visible(False) for x in ax.spines.values()), maxlen=0)

    # make ticks if necessary
    ax.tick_params(axis='y', which='both', labelsize=8, direction="out")
    ax.yaxis.tick_left()
    plt.yticks(y_ticks[0], y_ticks[1])

    if x_labels:
        ax.tick_params(axis='x', which='both', labelsize=8, direction="out")
        ax.xaxis.tick_bottom()
        plt.xticks(x, x_labels, rotation='vertical')
    else:
        plt.xticks([], [])

    # limits
    if end_gaps:
        gap = -gap
    ax.set_xlim(-1. * widths[0] / 2. + gap / 2., widths.sum() - widths[0] / 2. - gap / 2.)
    ax.set_ylim(0, y_ticks[0][-1])  # np.max(data_stack))

    # labels
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)


if __name__ == '__main__':
    demo()
