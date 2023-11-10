#!/usr/bin/env python3
"""
========
Overview
========
This script takes in a cumulative displacement file in .h5 format and
    - estimates ramp coefficients per epoch and model the coef time series with a linear + seasonal model
    - calculates temporal residuals from modelling ramp coef time series
    - calculates std of spatial residuals from deramped displacements
    - weight the inversion for linear + seasonal components from the time series
    - calculate standard error of model parameters from reduced chi-squares and covariance matrix
    - optionally export time series with seasonal component removed

Input:
    - [cum.h5] any .h5 file with a 3D array and a imdate vector

Outputs:
    - xx.h5.png [--plot_cum]
    - xx.h5_vel [-l] and .png [--plot_png]
    - xx.h5_vstd [-l] and .png [--plot_png]
    - xx.h5_amp [-s] and .png [--plot_png]
    - xx.h5_dt [-s] and .png [--plot_png]
    - xx.h5_ramp_coefs_resid_flat_std.png [-p]
    - xx.de_seasoned.h5 [--de_season]

"""

import numpy as np
import argparse
import LiCSBAS_io_lib as io_lib
import os
import matplotlib.pyplot as plt

# changelog
ver = "1.0"; date = 20230815; author = "Qi Ou, ULeeds"

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    """
    pass


def init_args():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-i', dest='infile', type=str, help="input float file")
    parser.add_argument('-p', dest='parfile', type=str, help="input slc.mli.par file")
    parser.add_argument('-o', dest='outfile', type=str, help="output float file")
    args = parser.parse_args()


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


def read_length_width():
    global length, width
    # read ifg size
    mlipar = os.path.join(args.parfile)
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))


if __name__ == "__main__":
    init_args()
    print(args.infile)
    read_length_width()
    print(length, width)
    data = io_lib.read_img(args.infile, length, width)
    plane_fit, range_coef, azi_coef = fit_plane(data)
    flatten = data - plane_fit


    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(data)
    # ax[1].imshow(plane_fit)
    # ax[2].imshow(flatten)
    # plt.savefig("flatten.png")

    flatten.astype(np.float32).tofile(args.outfile)






