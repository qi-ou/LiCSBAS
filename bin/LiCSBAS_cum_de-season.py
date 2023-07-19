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


def plot_cum_grid(cum, titles, suptitle, png, vmin=-20, vmax=20):
    # decide dimension of plotting grid
    n_im = cum.shape[0]
    n_row = int(np.sqrt(n_im))
    n_col = int(np.ceil(n_im / n_row))

    fig, ax = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(2*n_row*length/width, 2*n_col))
    for i in np.arange(n_im):
        row = i // n_col
        col = i % n_col
        print(i, row, col)
        im = ax[row, col].imshow(cum[i, :, :], vmin=vmin, vmax=vmax, cmap=SCM.roma.reversed())
        ax[row, col].set_title(titles[i])
    plt.suptitle(suptitle)
    fig.colorbar(im, ax=ax, label="Displacement, mm/yr")
    plt.savefig(png, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    start()
    init_args()

    # read input cum.h5
    cumh5 = h5.File(args.cumfile,'r')
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    n_im, length, width = cum.shape

    # read dt and amp from inputs
    delta_t = np.fromfile(args.delta_t, dtype=np.float32).reshape(length, width)
    amp = np.fromfile(args.amp, dtype=np.float32).reshape(length, width)

    cum = cum[:, ::args.downsample, ::args.downsample]
    delta_t = delta_t[::args.downsample,::args.downsample]
    amp = amp[::args.downsample,::args.downsample]
    length = length // args.downsample
    width = width // args.downsample

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt) - imdates_dt[0]) / 365.25)

    # remove seasonal_cum from cum to get remaining cum
    print("Removing seasonal component...")

    seasonal_cum = np.zeros((n_im, length, width)) * np.nan
    remain_cum = np.zeros((n_im, length, width)) * np.nan
    print("New cubes created...")
    for x in np.arange(width):
        if x % (width//10) == 0:
            print("Processing {}0%".format(x // (width//10)))
        for y in np.arange(length):
            seasonal_cum[:, y, x] = amp[y, x]*np.cos(2*np.pi*(dt_cum - delta_t[y, x]/365.26))
            remain_cum[:, y, x] = cum[:, y, x] - seasonal_cum[:, y, x]

            # plot time series de-seasoning
            # plt.plot(cum[:, y, x], label='cum')
            # plt.plot(seasonal_cum[:, y, x], label='seasonal')
            # plt.plot(remain_cum[:, y, x], label='remain')
            # plt.legend()
            # plt.show()

    # plot 3 cumulative displacement grids
    print("Plotting time series tiles...")
    plot_cum_grid(cum, imdates, "Cumulative displacement", args.cumfile + ".png")
    plot_cum_grid(seasonal_cum, imdates, "Seasonal cumulative displacement", args.cumfile + ".seasonal.png")
    plot_cum_grid(remain_cum, imdates, "De-seasoned cumulative displacement", args.cumfile + ".de-seasoned.png")

    finish()