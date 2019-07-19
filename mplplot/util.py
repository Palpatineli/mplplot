from typing import Tuple, Union, Optional, List, Dict
from matplotlib.colors import LinearSegmentedColormap
import re
from math import pi
import numpy as np

def get_ellipse(axes: np.ndarray, resolution: int=21) -> np.ndarray:
    """get polygonal ellipses for params in numpy arrays
    Args:
        axes: np.array([[a, b], ...])
            a - horizontal axis of the ellipse, in visual degrees
            b - vertical axis of the ellipse, in visual degrees
        resolution: number of polygons in the ellipse
    Returns:
        a 2d array, each row is an array of complex that forms the circumvent
    """
    angles = np.exp(-1j * np.linspace(-pi, pi, resolution + 1, endpoint=True))
    ray = np.outer((axes[:, 0] + axes[:, 1]) * 0.5, angles) + \
        np.outer((axes[:, 0] - axes[:, 1]) * 0.5, np.flipud(angles))
    return ray

def hex2rgb(hex: str) -> Tuple[int, ...]:
    hex = hex.lower()
    if not hasattr(hex2rgb, "re"):
        hex2rgb.re = re.compile("^#?(?:0x)?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})?")  # type: ignore
    match = hex2rgb.re.match(hex)  # type: ignore
    if match is None:
        raise ValueError(f"color hex formatted incorrectly: {hex}")
    return tuple([int(x, 16) for x in match.groups() if x is not None])

def rgb2hex(*rgb: int) -> str:
    if any(x > 255 or x < 0 for x in rgb):
        raise ValueError("rgb values out of range")
    if len(rgb) == 3:
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"
    elif len(rgb) == 4:
        r, g, b, a = rgb
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    else:
        raise ValueError("({}) cannot be converted to hex".format(", ".join([str(x) for x in rgb])))

def get_gradient_cmap(*colors: Union[str, Tuple[int, ...]], scale: Optional[List[float]] = None,
                      name: Optional[str] = None) -> LinearSegmentedColormap:
    colors = [(hex2rgb(color) if isinstance(color, str) else color) for color in colors]
    if scale is None:
        scale = np.linspace(0, 1.0, len(colors))
    if all(len(a) == 4 for a in colors):
        comp_names = ["red", "green", "blue", "alpha"]
    else:
        comp_names = ["red", "green", "blue"]
    cdict: Dict[str, List[List[float]]] = {x: list() for x in comp_names}
    for color, loc in zip(colors, scale):
        for idx, comp_name in enumerate(comp_names):
            cdict[comp_name].append([loc, color[idx] / 256, color[idx] / 256])
    return LinearSegmentedColormap(("new_cmap" if name is None else name), cdict)

def test_hex2rgb():
    assert hex2rgb("#619cff") == (97, 156, 255)
    assert hex2rgb("0x00ba38") == (0, 186, 56)
    try:
        hex2rgb("0x00g9a1")
        raise AssertionError("0x00g9a1 should not be parsed as hex!")
    except ValueError:
        pass
    assert hex2rgb("66b3aa") == (102, 179, 170)

def test_rgb2hex():
    assert rgb2hex(97, 156, 255) == "#619cff"
    assert rgb2hex(0, 186, 56) == "#00ba38"
    try:
        rgb2hex(-2, 13, 55)
        raise AssertionError("rgb2hex(-2, 13, 55) should be out of range!")
    except ValueError:
        pass
    try:
        rgb2hex(2, 258, 55)
        raise AssertionError("rgb2hex(-2, 13, 55) should be out of range!")
    except ValueError:
        pass
