##
from typing import Tuple, Union, List, Any, Sequence, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from matplotlib.pyplot import Axes
from matplotlib import pyplot as plt, ticker, rc
from matplotlib.colorbar import Colorbar
from matplotlib.text import Text
from matplotlib.image import AxesImage
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

def tsplot(ax: Axes, data: np.ndarray, time: np.ndarray = None, ci: float = 0.05, **kwargs) -> Axes:
    """Time series plot with mean and bootstrapped spread.
    Args:
        ax: Axes to draw on
        data: 2D array with rows of time series
        time: if time is not None, then used for x-axis, same length as data.shape[1]
        ci: confidence interval for two-ended p
    Returns:
        returns ax for chaining
    """
    if time is None or time.ndim != 1 or time.size < data.shape[1]:
        time = np.arange(data.shape[1])
    elif time.size > data.shape[1]:
        time = time[0: data.shape[1]]
    ave = data.mean(0)
    lower, upper = _bootstrap(data, ci=ci)
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

@dataclass
class Array:
    values: np.ndarray
    axes: List[np.ndarray]  # (column axis, row axis)

def labeled_heatmap(ax: Axes, data: Array, **kwargs):
    y_label, x_label = data.axes
    y_size, x_size = data.values.shape
    y_ticks, x_ticks = np.arange(0, y_size, y_size // 5), np.arange(0, x_size, x_size // 5)
    ax.set_xticks(x_ticks)
    if isinstance(x_label[0], str):
        ax.set_xticklabels(x_label)
    else:
        ax.set_xticklabels(x_label[x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_label[y_ticks])
    ax.imshow(data.values, **kwargs)

def heatmap(ax: Axes, data: Array, colorbar: Dict[str, Any] = {}, **kwargs) -> Tuple[AxesImage, Colorbar]:
    """Create a heatmap from a numpy array and two lists of labels.
    Args: 
        data: A 2D numpy array of shape (N, M).
        row_labels: A list or array of length N with the labels for the rows.
        col_labels: A list or array of length M with the labels for the columns.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
    Returns:
        the image
    """
    im = ax.imshow(data.values, **kwargs)  # Plot the heatmap
    # Create colorbar
    colorbar_label = colorbar.pop("label", None)
    colorbar_handle = ax.figure.colorbar(im, ax=ax, **colorbar)
    if colorbar_label is not None:
        colorbar_handle.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")
    (y_size, x_size), (y_label, x_label) = data.values.shape, data.axes
    ax.set_xticks(np.arange(x_size))
    ax.set_yticks(np.arange(y_size))
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(x_size + 1) - .5, minor=True)
    ax.set_yticks(np.arange(y_size + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, colorbar_handle

def annotate_heatmap(im: AxesImage, data: Optional[np.ndarray] = None, valfmt: Union[str, ticker.Formatter] = "{x:.2f}",
                     textcolors: Tuple[str, str] = ("white", "black"), threshold: Optional[float] = None,
                     **textkw) -> List[Text]:
    """Annotate an existing heatmap.
    Args:
        im: The AxesImage to be labeled.
        data: Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt: The format of the annotations inside the heatmap. Formatting string or ticker.Formatter
        textcolors: a tuple of two text colors for dark and light background
        threshold: Value in data units according to which the colors from textcolors are applied. Middle of the colormap is used if None.
        **kwargs: All other arguments are forwarded to each call to `text`
    Returns:
        a list of Text objects
    """
    if data is None:
        data = im.get_array()
    threshold = im.norm(data.max())/2 if threshold is None else im.norm(threshold)  # norm is from colors.Normalize instance
    textkw.setdefault("horizontalalignment", "center")
    textkw.setdefault("verticalalignment", "center")
    if isinstance(valfmt, str):  # Get the formatter in case a string is supplied
        valfmt = ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):  # Loop over the data and create a `Text` for each "pixel".
        for j in range(data.shape[1]):
            textkw["color"] = textcolors[int(im.norm(data[i, j]) > threshold)]  # Change the text's color
            texts.append(im.axes.text(j, i, valfmt(data[i, j], None), **textkw))
    return texts

##
def test_heatmap():
    a = np.arange(25).reshape(5, 5)
    b = Array(a, [np.arange(5) / 5.0, ['this', 'that', 'crap', 'shit', 'okay']])
    from mplplot import Figure
    with Figure() as axes:
        ax = axes[0]
        rc('font', family='Lato', weight='bold', size=12)
        im, cb = heatmap(ax, Array(a, [np.arange(5) / 5.0, ['this', 'that', 'crap', 'shit', 'okay']]))
        annotate_heatmap(im, valfmt="{x:.1f}")
