from math import pi
import numpy as np
import matplotlib.pyplot as plt

def degree_to_rad(angle_degree):
    return (angle_degree * pi) / 180.


def rad_to_degree(angle_rad):
    return (angle_rad * 180.) / pi


def get_close_degrees(arr, index, degree_dist=20):
    """
    ------
    INPUTS
    arr: np.array, values of degrees between
    index: int, the value of the array to index at
    degree_dist: int, value of how many degrees there can be between
                 the points to count.

    """
    #
    boolean = [np.isclose(arr[index], arr, .1, degree_dist)]
    return arr[boolean], boolean


def count_45s(arr):
    """
    Go through every 45 degrees of a circle, and count the number
    of events in the array that fall within 20 degrees of the arr
    -----
    INPUTS:
    arr: np.array, an array of your degrees to check over
    -----
    outputs:
    dd: dict, keys are the degrees (e.g. 45), the values are the #
    """
    d = {}
    # Get every 45th degree in 360
    points = np.arange(-360, 360 + 45, 45)
    # Check if it's within 20 degrees, add to d the number of them
    for degree in points:
        d[degree] = arr[np.isclose(degree, arr, .1, 20)].shape[0]

    for k, v in enumerate(sorted(d.keys())):
        convert = 360 + v
        d[convert] = d[convert] + d[v]
        if v == -45:
            break
    dd = {v: d[v] for k, v in enumerate(d) if v >= 0}
    test = zip(degree_to_rad(np.array(dd.keys())), dd.values())
    return dd, test


def polar_chart(ts, electrode):
    """
    ts: timeseries
    electrode: int
    """

    a, b = count_45s(rad_to_degree(ts[electrode, 0, :, 5].data))
    """
    =======================
    Pie chart on polar axis
    =======================

    Demo of bar plot on a polar axis.
    """

    # Compute pie slices
    t = [0.0, 3.9269908169872414, 2.3561944901923448, 6.2831853071795862, 0.78539816339744828,
         4.7123889803846897, 3.1415926535897931, 1.5707963267948966, 5.497787143782138]
    N = len(t)
    # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    counts = [bb[1] for bb in b]
    radii = np.array(counts)
    # width = np.ones(N)/7

    ax = plt.subplot(111, projection='polar')

    bars = ax.bar(counts, t)
    # bars = ax.bar(t, radii, width=width,)

    # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    # bar.set_facecolor(plt.cm.jet(r / 10.))
    # bar.set_alpha(0.5)

    plt.show()