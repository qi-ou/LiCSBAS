#!/usr/bin/env python3
"""
========
Overview
========
This script removes the seasonal components from a time series cube and plot the de-seasoned displacement series in a grid.
"""

import numpy as np
import h5py as h5
import datetime as dt
import matplotlib.pyplot as plt
import argparse
import SCM
import time
import os
import sys

# changelog
ver = "1.0"; date = 20230605; author = "Qi Ou, ULeeds"  # removes the seasonal components from a time series cube and plot


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
    parser.add_argument('-i', dest='cumfile', default="cum.h5", type=str, help="input .h5 file")
    parser.add_argument('-t', dest='delta_t', default="xx.dt", type=str, help="this is an output of LiCSBAS_cum2vel.py")
    parser.add_argument('-a', dest='amp', default="xx.amp", type=str, help="this is an output of LiCSBAS_cum2vel.py")
    parser.add_argument('-d', dest='downsample', default=10, type=int, help="downsample cumfile before removing seasonal component")
    parser.add_argument('-s', dest='de_season', default=False, action='store_true', help="remove seasonal component")
    parser.add_argument('-p', dest='deramp', default=False, action='store_true', help="remove planar ramp")
    parser.add_argument('-r', dest='ref', default=False, action='store_true', help="reference to the center of the image")
    parser.add_argument('--heading', type=float, default=0, choices=[-10, -170, 0], help="heading azimuth, -10 for asc, -170 for dsc, 0 if in radar coordinates, required if using deramp")

    args = parser.parse_args()


def start():
    global start_time
    start_time = time.time()
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minite, sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    # print('Output: {}\n'.format(os.path.relpath(args.outfile)))


def plot_cum_grid(cum1, titles, suptitle, png):
    print("Plotting {}".format(png))
    # decide dimension of plotting grid
    n_im = cum1.shape[0]
    n_row = int(np.sqrt(n_im))
    n_col = int(np.ceil(n_im / n_row))

    if args.ref:
        cum = np.ones(cum1.shape) * np.nan
        for i in np.arange(n_im):
            cum[i, :, :] = cum1[i, :, :] - cum1[i, cum1.shape[1]//2, cum1.shape[2]//2]
        suptitle = suptitle + "_ref2center"
    else:
        cum = cum1

    vmin_list = []
    vmax_list = []
    for i in np.arange(n_im):
        vmin_list.append(np.nanpercentile(cum[i, :, :], 1))
        vmax_list.append(np.nanpercentile(cum[i, :, :], 99))
    vmin = min(vmin_list)
    vmax = max(vmax_list)

    fig, ax = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(2*n_col*width/length, 2*n_row))
    for i in np.arange(n_im):
        row = i // n_col
        col = i % n_col
        # print(i, row, col)
        im = ax[row, col].imshow(cum[i, :, :], vmin=vmin, vmax=vmax, cmap=SCM.roma.reversed())
        ax[row, col].set_title(titles[i])
    plt.suptitle(suptitle)
    fig.colorbar(im, ax=ax, label="Displacement, mm/yr")
    plt.savefig(png, bbox_inches='tight')
    plt.close()


def fit_plane(z, theta=0):
    """Fit a plane to data.
    Parameters
    ----------
    z : `numpy.ndarray`
        2D array of z values
    theta : heading angle in radian
    """
    yy, xx = np.indices(z.shape)
    yy = -yy  # minus to make y axis positive upward, otherwise indices increases down the rows
    ones = np.ones(z.shape)

    pts = np.isfinite(z)
    coefs = np.linalg.lstsq(np.stack([xx[pts], yy[pts], ones[pts]]).T, z[pts].flatten(), rcond=None)[0]

    plane_fit = coefs[0] * xx + coefs[1] * yy + coefs[2]

    # rotate axis
    range_coef = coefs[0] * np.cos(theta) + coefs[1] * np.sin(theta)
    azi_coef = coefs[0] * -np.sin(theta) + coefs[1] * np.cos(theta)
    return plane_fit, range_coef, azi_coef


if __name__ == "__main__":
    start()
    init_args()

    # read input cum.h5
    cumh5 = h5.File(args.cumfile,'r')
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    n_im, length, width = cum.shape

    # downsample
    cum = cum[:, ::args.downsample, ::args.downsample]
    plot_cum_grid(cum, imdates, args.cumfile, args.cumfile + ".png")

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt) - imdates_dt[0]) / 365.25)
    epochs = ([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates])

    if args.de_season:
        # read dt and amp from inputs and downsample
        delta_t = np.fromfile(args.delta_t, dtype=np.float32).reshape(length, width)
        amp = np.fromfile(args.amp, dtype=np.float32).reshape(length, width)
        delta_t = delta_t[::args.downsample, ::args.downsample]
        amp = amp[::args.downsample, ::args.downsample]

        # remove seasonal_cum from cum to get remaining cum
        print("Removing seasonal component...")

        seasonal_cum = np.zeros(cum.shape) * np.nan
        remain_cum = np.zeros(cum.shape) * np.nan
        print("New cubes created...")
        for x in np.arange(cum.shape[2]):
            if x % (cum.shape[2]//10) == 0:
                print("Processing {}0%".format(x // (cum.shape[2]//10)))
            for y in np.arange(cum.shape[1]):
                seasonal_cum[:, y, x] = amp[y, x]*np.cos(2*np.pi*(dt_cum - delta_t[y, x]/365.26))
                remain_cum[:, y, x] = cum[:, y, x] - seasonal_cum[:, y, x]
        # plot cumulative displacement grids
        plot_cum_grid(seasonal_cum, imdates, "Seasonal {}".format(args.cumfile), args.cumfile + ".seasonal.png")
        plot_cum_grid(remain_cum, imdates, "De-seasoned {}".format(args.cumfile), args.cumfile + ".de-seasoned.png")

    if args.deramp:
        ramp_cum = np.zeros(cum.shape) * np.nan
        range_coefs = []
        azi_coefs = []
        for i in np.arange(n_im):
            plane_fit, range_coef, azi_coef = fit_plane(cum[i, :, :], np.deg2rad(args.heading))
            ramp_cum[i, :, :] = plane_fit
            range_coefs.append(range_coef)
            azi_coefs.append(azi_coef)

        # plot time series of ramp parameters
        plt.plot(epochs, range_coefs, label="range_coef")
        plt.plot(epochs, azi_coefs, label="azi_coef")
        plt.xlabel("Epoch")
        plt.ylabel("ramp rate unit/pixel")
        plt.title(args.cumfile)
        plt.legend()
        plt.savefig(args.cumfile + "_ramp_coefs.png")
        plt.close()

        flat_cum = cum - ramp_cum
        ramp_cum[np.isnan(cum)] = np.nan
        plot_cum_grid(ramp_cum, imdates, "Best-fit ramps {}".format(args.cumfile), args.cumfile + "_ramps.png")
        plot_cum_grid(flat_cum, imdates, "Flattened {}".format(args.cumfile), args.cumfile + "_flattened.png")

    finish()