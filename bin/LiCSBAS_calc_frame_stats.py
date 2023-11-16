#!/usr/bin/env python3
"""
========
Overview
========
This script:
 - calculates the block sum of unw pixels
 - calculates the block sum of coherence
 - calculates the block sum of connected component size
 - calculates the block std of height
 - combine and normalise a proxy [0-1] of suitability of reference window
 - choose amongst the selected windows (above threshold) the nearest to desired reference location
 - discard ifgs with all nan values in the chosen reference window

===============
Input & output files
===============

Inputs in frameID/:
- final_list_bt_stats.txt containing columns of IFG bp bt pix_cov avg_coh extracted from 11ifg_stats.txt [e.g. 20141029_20141122 33.2 24 1.000 0.387]
Outputs in frameID/ :
- frameID_histogram.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import sys
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

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
    parser.add_argument('-d', dest="frame_dir", default="./", help="master directory containing all frame folders")
    parser.add_argument('-l', dest='ifg_list', default="final_list.txt", help="ifg list as a subset of ifgs in TS_*/11ifg_stats.txt")
    args = parser.parse_args()

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20231116; author="Qi Ou"
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
    print('Output directory: {}\n'.format(os.path.relpath(args.frame_dir)))


# Function to process each final_list_stats.txt file
def txt2hist(frameID_path, file):
    # Read the txt file into a pandas dataframe
    df = pd.read_csv(file, delimiter='\s+', header=None, usecols=[args.column], names=['value'])

    # Plot the histogram of the specific column
    bin_width = 6
    bin_edges = np.arange(min(df['value']), max(df['value']) + bin_width, bin_width) - bin_width / 2

    plt.hist(df['value'], bins=bin_edges, color='blue', edgecolor='black')
    plt.title(os.path.basename(frameID_path))  # Use the folder name as the plot title
    plt.xlabel(args.xlabel)
    plt.ylabel('Number of IFGs')
    # Annotate the histogram with mean and standard deviation
    mean_val = df['value'].mean()
    std_dev = df['value'].std()
    plt.annotate(f'NumIFGs: {len(df):.2f}\nMean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}', xy=(0.7, 0.8),
                 xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white"))
    print(frameID, min(df['value']), max(df['value']),  f'{mean_val:.2f}', f'{std_dev:.2f}')
    # Save the histogram as a PNG file named after the folder
    output_file_name = os.path.join(frameID_path, os.path.basename(frameID_path) + '_histogram.png')
    plt.savefig(output_file_name)
    plt.close()  # Close the plot to avoid displaying it


def calc_epoch_number_from_df(df):
    # Annotate the histogram with mean and standard deviation
    df['epoch1'] = df['ifg'].str.slice(0, 8)
    df['epoch2'] = df['ifg'].str.slice(-8)
    epochs = set(df['epoch1']).union(set(df['epoch2']))
    epoch_num = len(epochs)
    return epoch_num


# Function to process each final_list_stats.txt file
def plot_stats(frameID_path, stats_file, ifg_list_file):
    # Read the txt file into a pandas dataframe
    df = pd.read_csv(stats_file, delimiter='\s+', header=None, comment='#', usecols=[0, 1, 2, 3, 4],
                     names=['ifg', 'bp', 'bt', 'pix_cov', 'avg_coh'],
                     dtype={'ifg': str, 'bp': float, 'bt': int, 'pix_cov': float, 'avg_coh': float})

    # Annotate the histogram with mean and standard deviation
    epoch_number = calc_epoch_number_from_df(df)

    final_list = pd.read_csv(ifg_list_file, header=None, names=["ifg"])
    final_df = df[df["ifg"].isin(final_list["ifg"])]
    final_epoch_number = calc_epoch_number_from_df(final_df)

    # Scatter plot with size and color specified by columns
    plt.scatter(df['bt'], df['avg_coh'], label="start_list")
    plt.scatter(final_df['bt'], final_df['avg_coh'], label="final_list")

    # Add labels and title
    plt.legend()
    plt.xlabel('Temporal Baseline / Days')
    plt.ylabel('Average Coherence')
    plt.title(os.path.basename(frameID_path))

    # Calculate summary statistics for numeric columns, Round the summary statistics to one decimal place
    df_stats = df.describe().round(1)
    final_df_stats = final_df.describe().round(1)

    plt.annotate('IFGs: {:.0f}/{:.0f}\nEpochs:{:d}/{:d}\nBt:  {:.0f}/{:.0f}\nCoh: {:.2f}/{:.2f}'
                 .format(final_df_stats["bt"].loc["count"], df_stats["bt"].loc["count"],
                         final_epoch_number, epoch_number,
                         final_df_stats["bt"].loc["mean"], df_stats["bt"].loc["mean"],
                         final_df_stats["avg_coh"].loc["mean"], df_stats["avg_coh"].loc["mean"]), xy=(0.6, 0.8),
                 xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white"))

    print(os.path.basename(frameID_path), "{:.0f} {:.0f} {:d} {:d} {:.0f} {:.0f} {:.2f} {:.2f}".format(
        final_df_stats["bt"].loc["count"], df_stats["bt"].loc["count"],
        final_epoch_number, epoch_number,
        final_df_stats["bt"].loc["mean"], df_stats["bt"].loc["mean"],
        final_df_stats["avg_coh"].loc["mean"], df_stats["avg_coh"].loc["mean"]
    ))
    # Save the histogram as a PNG file named after the folder
    output_file_name = os.path.join(frameID_path, os.path.basename(frameID_path) + '_frame_stats.png')
    plt.savefig(output_file_name)
    plt.close()  # Close the plot to avoid displaying it


def main():
    start()
    init_args()
    finish()


if __name__ == "__main__":
    # main()
    start()
    init_args()

    # Loop through all frameID folders
    for frameID in os.listdir(args.frame_dir):
        frameID_path = os.path.join(args.frame_dir, frameID)
        stats_file = os.path.join(args.frame_dir, frameID, "TS_GEOCml10GACOS/info/11ifg_stats.txt")
        ifg_list_file = os.path.join(args.frame_dir, frameID, "final_list.txt")
        plot_stats(frameID_path, stats_file, ifg_list_file)

    finish()
