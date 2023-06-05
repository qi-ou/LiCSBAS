#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from osgeo import gdal
from scipy import stats
import SCM

class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file."""

    def __init__(self, filename):
        self.ds = gdal.Open(filename)
        self.basename = os.path.splitext(os.path.basename(filename))[0]
        self.band = self.ds.GetRasterBand(1)
        self.data = self.band.ReadAsArray()
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize
        self.left = self.ds.GetGeoTransform()[0]
        self.top = self.ds.GetGeoTransform()[3]
        self.xres = self.ds.GetGeoTransform()[1]
        self.yres = self.ds.GetGeoTransform()[5]
        self.right = self.left + self.xsize * self.xres
        self.bottom = self.top + self.ysize * self.yres
        self.projection = self.ds.GetProjection()
        pix_lin, pix_col = np.indices((self.ds.RasterYSize, self.ds.RasterXSize))
        self.lat, self.lon = self.top + self.yres * pix_lin, self.left + self.xres * pix_col


if __name__ == "__main__":
    # parse frame name as argument
    parser = argparse.ArgumentParser(description="Measuring slope of middle column")
    parser.add_argument("tif", type=str, help="tif file")
    parser.add_argument("angle", type=float, help="angle in degree, where 0 = vertical; positive means clockwise rotation about midpoint")
    args = parser.parse_args()

    tif = OpenTif(args.tif)
    title = args.tif.split('/')[-1][:17]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    im=ax[0].imshow(tif.data, vmin=np.nanpercentile(tif.data, 0.5), vmax=np.nanpercentile(tif.data, 99.5), cmap=SCM.roma.reversed())
    ax[0].set_title(title)
    plt.colorbar(im, ax=ax[0])

    column_latitudes = tif.lat[:, 0]
    ys = np.arange(tif.ysize)
    xs = np.array([int((tif.ysize // 2 - y) * np.tan(np.deg2rad(args.angle)) + tif.xsize // 2) for y in ys])
    zs = tif.data[ys, xs]
    non_nan_mask = ~np.isnan(zs)
    slope, intercept, r_value, p_value, std_err = stats.linregress(column_latitudes[non_nan_mask], zs[non_nan_mask])

    ax[0].plot(xs, ys, c='k', linewidth=1)
    ax[1].plot(column_latitudes, zs)
    ax[1].annotate("Phase = {:.2f} latitude + {:.2f} \n$R^2$ = {:.3f}".format(slope, intercept, r_value ** 2),
                   xy=(0.05, 0.965),
                   xycoords='axes fraction', ha='left', va='top', zorder=10)
    ax[1].plot(column_latitudes, column_latitudes * slope + intercept)
    ax[1].set_xlabel("Latitude")
    # ax[1].set_ylabel("Phase / mm")
    ax[1].set_title("Middle Column")
    plt.tight_layout()
    plt.savefig("{}.middle_column_slope.png".format(title), dpi=500)
    plt.close()

    print(title[:17], slope)