#!/usr/bin/env python3
"""
========
Overview
========
This script loads epoch-wide tif files into an h5 cube referenced to the first epoch, and copy over other metadata from an existing cum.h5
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import sys
from matplotlib import cm
import SCM
import glob
from osgeo import gdal
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib


global ver, date, author
ver = "1.0"; date = 20230630; author = "Qi Ou, ULeeds"  # Difference two rasters


class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file."""
    def __init__(self, filename, sigfile=None, incidence=None, heading=None, N=None, E=None, U=None):
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

        # convert 0 and 255 to NaN
        self.data[self.data==0.] = np.nan


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


def init_args():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument(dest='first', type=str, help="path to first raster (binary or tiff)")
    parser.add_argument(dest='second', type=str, help="path to second raster of the same size (binary or tiff)")
    parser.add_argument('--vmin', dest='vmin', type=float, help="lower end of colourmap")
    parser.add_argument('--vmax', dest='vmax', type=float, help="upper end of colourmap")
    parser.add_argument('-o', dest='outfile', type=str, help="pngfile of a 3 panel plot showing the difference")
    parser.add_argument('-d', dest='ifgdir', default='GEOCml10GACOS', type=str, help="directory containing slc.mli.par, required if rasters not in tiff format")
    args = parser.parse_args()


def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish(outfile):
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output: {}\n'.format(os.path.relpath(outfile)))


if __name__ == "__main__":
    init_args()
    start()

    mlipar = os.path.join(args.ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    array1 = io_lib.read_img(args.first, length, width)
    array2 = io_lib.read_img(args.second, length, width)
    title1= os.path.splitext(os.path.basename(args.first))[0]
    title2= os.path.splitext(os.path.basename(args.second))[0]

    data3 = [array1, array2, array1-array2]
    title3 = [title1, title2, "Diff (panel 1 - 2)"]
    if args.outfile:
        pnefile=args.outfile
    else:
        pngfile = "diff_{}-{}.png".format(title1, title2)

    if args.vmin:
        vmin = args.vmin
    else:
        vmin = None

    if args.vmax:
        vmax = args.vmax
    else:
        vmax = None

    plot_lib.make_3im_png(data3, pngfile, SCM.roma.reversed(), title3, vmin=vmin, vmax=vmax, cbar=True)

    finish(pngfile)
