from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import numpy as np
wes_palettes = {
  'BottleRocket1' : ["#A42820", "#5F5647", "#9B110E", "#3F5151", "#4E2A1E", "#550307", "#0C1707"],
  'BottleRocket2' : ["#FAD510", "#CB2314", "#273046", "#354823", "#1E1E1E"],
  'Rushmore1' : ["#E1BD6D", "#EABE94", "#0B775E", "#35274A" ,"#F2300F"],
  'Rushmore' : ["#E1BD6D", "#EABE94", "#0B775E", "#35274A" ,"#F2300F"],
  'Royal1' : ["#899DA4", "#C93312", "#FAEFD1", "#DC863B"],
  'Royal2' : ["#9A8822", "#F5CDB4", "#F8AFA8", "#FDDDA0", "#74A089"],
  'Zissou1' : ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"],
  'Darjeeling1' : ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"],
  'Darjeeling2' : ["#ECCBAE", "#046C9A", "#D69C4E", "#ABDDDE", "#000000"],
  'Chevalier1' : ["#446455", "#FDD262", "#D3DDDC", "#C7B19C"],
  'FantasticFox1' : ["#DD8D29", "#E2D200", "#46ACC8", "#E58601", "#B40F20"],
  'Moonrise1' : ["#F3DF6C", "#CEAB07", "#D5D5D3", "#24281A"],
  'Moonrise2' : ["#798E87", "#C27D38", "#CCC591", "#29211F"],
  'Moonrise3' : ["#85D4E3", "#F4B5BD", "#9C964A", "#CDC08C", "#FAD77B"],
  'Cavalcanti1' : ["#D8B70A", "#02401B", "#A2A475", "#81A88D", "#972D15"],
  'GrandBudapest1' : ["#F1BB7B", "#FD6467", "#5B1A18", "#D67236"],
  'GrandBudapest2' : ["#E6A0C4", "#C6CDF7", "#D8A499", "#7294D4"],
  'IsleofDogs1' : ["#9986A5", "#79402E", "#CCBA72", "#0F0D0E", "#D9D0D3", "#8D8680"],
  'IsleofDogs2' : ["#EAD3BF", "#AA9486", "#B6854D", "#39312F", "#1C1718"]}


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def create_colormap(colors=None, cmap_name=None, bins=10):
    """

    Parameters
    ----------
    colors
    cmap_name
    bins

    Returns
    -------

    """
    if colors is None:
        # colors = ['#cb654f', '#d3b1a7', '#cfcb9c', '#8cbea3', '#dfba47',
        # '#fad77b', '#cdc08c','#9c964a', '#f4b5bd', '#86d4e4']
        colors = ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6",
                  "#E1BD6D", "#EABE94", "#0B775E", "#35274A", "#F2300F"]
        if cmap_name is None:
            # cmap_name = 'MoonriseKingdomCustom'
            cmap_name = 'Rushmore1'
    if cmap_name is None:
        cmap_name = 'my_list'

    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=bins)
    return cm

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def discrete_cmap(N=8):
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',
              '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',
              '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]
    cmap3 = col.ListedColormap(cpool[0:N], 'indexed')
    cm.register_cmap(cmap=cmap3)


def fill_between_stderr(arr):
    """
    arr: array of all subjects, axis 0 = subjects
    arr_tstat: array of plotting falls we'll use for the fillbetween
    """
    mean = np.nanmean(arr,axis=0)
    std_err_mean = np.nanstd(arr,axis=0) / ( np.sqrt( len(arr) ) )
    y1, y2 = np.array( (mean+std_err_mean) ), np.array( (mean-std_err_mean) )
    return y1, y2
    #plt.fill_between(y1=y1,y2=y2, x=x_axis, color = 'blue', alpha=.25)