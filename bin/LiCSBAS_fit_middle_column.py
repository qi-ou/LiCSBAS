#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from osgeo import gdal
from scipy import stats


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
        self.lat, self.lon = self.top + self.yres*pix_lin, self.left+self.xres*pix_col


if __name__ == "__main__":

    # parse frame name as argument
    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument("tif", help="tif file")
    args = parser.parse_args()

    tif = OpenTif(args.tif)
    title = args.tif.split('/')[-1]

    fig, ax=plt.subplots(1,2, figsize=(6, 3))
    ax[0].imshow(tif.data, vmin = np.nanpercentile(tif.data, 0.5), vmax = np.nanpercentile(tif.data, 99.5))
    ax[0].plot([tif.xsize // 2, tif.xsize // 2], [0, tif.ysize+1])
    ax[0].set_title(title)

    middle_column = tif.data[:, tif.xsize // 2]
    middle_column_latitudes = tif.lat[:, tif.xsize // 2]
    ax[1].plot(middle_column_latitudes, middle_column)
    non_nan_mask = ~np.isnan(middle_column)
    slope, intercept, r_value, p_value, std_err = stats.linregress(middle_column_latitudes[non_nan_mask], middle_column[non_nan_mask])

    ax[1].annotate("Phase = {:.2f} latitude + {:.2f} \n$R^2$ = {:.3f}".format(slope, intercept, r_value**2), xy=(0.05, 0.965),
                    xycoords='axes fraction', ha='left', va='top', zorder=10)
    ax[1].plot(middle_column_latitudes, middle_column_latitudes*slope+intercept)
    ax[1].set_xlabel("Latitude")
    ax[1].set_ylabel("Phase / mm")
    ax[1].set_title("Middle Column")
    plt.tight_layout()
    plt.savefig("{}.png".format(title), dpi=500)
    plt.close()

    print(title, slope)

