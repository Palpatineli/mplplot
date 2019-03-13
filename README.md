# MPLPlot

Plotting Functions including:

A resource manager adaptor for matplotlib

```python3
from mplplot.importer import MplFigure

with MplFigure(file_path, (size_x, size_y), despine={'bottom': false}) as ax:
    # plot with ax
```

Where seaborn style is auto initialized and figure is saved to path on exit, and extra kwargs are passed through to figure.subplots
