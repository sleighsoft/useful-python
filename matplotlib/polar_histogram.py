# Requires
# Python, numpy matplotlib
#
# Can be installed with:
# pip install numpy matplotlib
#
# Description:
# Plots a radial histogram where colors indicate the weight instead of height.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def polar_histogram(data, cmap='gist_heat_r'):

    figure, axes = plt.subplots(subplot_kw={'projection': 'polar', 'frameon': False}, figsize=[8,6])

    axes.set_rticks([])
    axes.set_thetagrids([], labels=[])
    axes.grid(False)

    histogram = np.histogram(data, bins=360)[0]

    rad = histogram
    azimut = np.linspace(0, 2*np.pi, 360)

    y, x = np.meshgrid(rad, azimut)
    _, z = np.meshgrid(azimut, rad)

    # Colormap: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    axes.pcolormesh(x, y, z, antialiased=True, cmap=cmap)

    return figure, axes
