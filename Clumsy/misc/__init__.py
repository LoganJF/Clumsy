try:
    from .matplotlibmisc import *
except ImportError as e:

    try:
        from .matplotlibmisc import wes_palettes, rgb_to_hex, hex_to_rgb
        print('succeeded to install wes_palettes, rgb_to_hex, hex_to_rgb from script')
        print('failed to install discrete_colormap, create_colormap, shiftedColorMap from script')

    except ImportError as e:
        warn = 'failed to install discrete_colormap, create_colormap, shiftedColorMap from script, wes_palettes, {}'
        print(warn.format('rgb_to_hex, hex_to_rgb from script'))

from .misc import *
from .inherit_docstrings import InheritableDocstrings

