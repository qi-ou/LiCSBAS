#!/usr/bin/env python3
"""
========
Overview
========
This script:
 - copies file 130cum*.h5 to cum.h5
 - copies folder 130results* to results
 - calcs n_unw, n_loop_err, coh_avg
 - assumbles all results into cum.h5

===============
Input & output files
===============

Inputs in GEOCml*/ (--comp_cc_dir):
 - slc.mli.par
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.cc

Inputs in GEOCml*/ (--unw_dir):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Inputs in TS_GEOCml*/ :
 - 130cum*.h5            : Cumulative displacement (time-seires) in mm from final iteration, to copy to cum.h5

 - 130results*           : Results from final iteration of automatic correction, to copy to results
    -vel[.png]           : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept[.png]    : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms[.png]     : RMS of residual in mm
   - n_gap[.png]         : Number of gaps in SB network
   - n_ifg_noloop[.png]  : Number of ifgs with no loop
   - maxTlen[.png]       : Max length of continous SB network in year

 - info
   - 120ref.txt          : Reference window

Outputs in TS_GEOCml*/ :
 - cum.h5                : Cumulative displacement (time-seires) in mm

 - results/
   - n_unw[.png]         : Number of available unwrapped data to be used
   - coh_avg[.png]       : Average coherence
   - n_loop_err[.png]    : Number of remaining loop errors (>pi) in data to be used

=====
Usage
=====
LiCSBAS133_write_h5.py [-h] [-f FRAME_DIR] [-c COMP_CC_DIR] [-t TS_DIR] [--suffix SUFFIX]
"""

#%% Change log
'''
v1.0 20220928 Qi Ou, Leeds Uni
'''

#%% Import
import os
import time
import shutil
import numpy as np
import h5py as h5
from pathlib import Path
import argparse
import sys
import re
import xarray as xr
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib


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
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output")
    parser.add_argument('-c', dest='comp_cc_dir', default="GEOCml10GACOS", help="folder containing connected components and coherence files")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs, if not given, all ifgs in -c are read")
    parser.add_argument('-o', dest='outfile', default=None, type=str, help="output filename")

    args = parser.parse_args()

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def set_input_output():
    global ccdir, ifgdir, ifgdates, coh_avgfile

    # define input directories and file
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))

    if args.ifg_list:
        ifgdates = io_lib.read_ifg_list(args.ifg_list)
    else:
        ifgdates = tools_lib.get_ifgdates(ccdir)

    if args.outfile:
        coh_avgfile = os.path.join(args.outfile)
    else:
        coh_avgfile = os.path.join('coh_avg')

def read_length_width():
    global length, width
    # read ifg size
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

def calc_coh_avg():
    print("Computing average coherence...")
    # calc n_unw and avg_coh of final data set
    coh_avg = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates:
        ccfile = os.path.join(ccdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
        else:
            coh = io_lib.read_img(ccfile, length, width)
            coh[np.isnan(coh)] = 0  # Fill nan with 0
        coh_avg += coh
        n_coh += (coh!=0)
    coh_avg[n_coh==0] = np.nan
    n_coh[n_coh==0] = 1 #to avoid zero division
    coh_avg = coh_avg/n_coh
    coh_avg[coh_avg==0] = np.nan

    ### Write to file
    coh_avg.tofile(coh_avgfile)

    ### Save png
    title = 'Average coherence'
    cmap_noise = 'viridis'
    plot_lib.make_im_png(coh_avg, coh_avgfile+'.png', cmap_noise, title)


def main():
    global ifgdates

    # intialise
    start()
    init_args()

    # directory settings
    set_input_output()
    read_length_width()

    # calc quality stats based on the final corrected unw
    calc_coh_avg()

    # report finish
    finish()


if __name__ == '__main__':
    main()
