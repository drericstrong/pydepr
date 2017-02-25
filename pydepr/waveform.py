# -*- coding: utf-8 -*-
"""
    pydepr.waveform
    ~~~~~~~~~~
    PyDePr is a set of tools for processing degradation models. This module
    contains tools for processing and validating waveforms, such as
    cylinder firing pressure curves.

    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import pyedna.ezdna as dna
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# "GLOBAL" VARIABLES
SITE = "MDSSCSC1"
# Format is equipment tag, number of cylinders, pressure curve tag identifier
PARAM_DICT = {"ANV": [["ADE1", 7, "PA"], ["ADE2", 7, "PA"], ["ADE3", 7, "PA"]],
              "CAL": [["ADE1", 7, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]],
              "FLV": [["ADE1", 6, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]],
              "HDV": [["ADE1", 7, "PC"], ["ADE2", 7, "PC"], ["ADE3", 7, "PC"]],
              "HRV": [["ADE1", 7, "PC"], ["ADE2", 7, "PC"], ["ADE3", 7, "PC"]],
              "LBV": [["ADE1", 6, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]],
              "LEV": [["ADE1", 6, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]],
              "MSV": [["ADE1", 6, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]],
              "ORV": [["ADE1", 7, "PC"], ["ADE2", 6, "PC"], ["ADE3", 6, "PC"]]
              # "PGV" : [["HVDG", 8, "PC"], ["LVG1", 6, "PC"], ["LVG2", 6, "PC"]],
              # "PLV" : [["HVDG", 8, "PC"], ["LVG1", 6, "PC"], ["LVG2", 6, "PC"]]
              }
CFP_SIZE = {"ANV": 720,
            "CAL": 7200,
            "FLV": 7200,
            "HDV": 7200,
            "HRV": 7200,
            "LBV": 7200,
            "LEV": 7200,
            "MSV": 7200,
            "ORV": 7200
            # "PGV" : 7200,
            # "PLV" : 7200
            }
SERVICE = "DD"
CFP_START_VAL = -7291290
PLOT_CFPS = True

# MATPLOTLIB CONFIG
sns.set_palette('colorblind')
font = {'size': 28}
mpl.rc('font', **font)
FSIZE = (12, 8)


# FUNCTIONS
def run_data(equip, num_cyls, pres_tag, cfp_size, cfp_service, start_date_,
             end_date_):
    # tag_data will contain pandas DataFrames for all cylinders per one equipment
    tag_data = []
    # The cylinder tags always have the same format: equipment ("ADE1") +
    # "C" + cylinder number + either "PA" or "PC" depending on ship
    for cyl_num in range(1, num_cyls + 1):
        tag = equip + "C" + str(cyl_num) + pres_tag
        full_tag = ".".join([SITE, cfp_service, tag])
        # Pull the raw data. It's very important to pull in "raw" mode, because
        # "snap" will not retrieve the full CFP curve, since multiple data
        # points may be written to the same timestamp
        df = dna.GetHistRaw(full_tag, start_date_, end_date_, high_speed=True)
        # Process the CFP curve. The start of each curve is identified by
        # looking for the CFP_START_VAL value; there is a better way to do
        # this, but it works for now.
        proc_df = process_cfp_curve(df, full_tag, cfp_size)
        # If the user wants, each CFP curve can be plotted. Warning- this will
        # take a large amount of disk space and time.
        if PLOT_CFPS:
            for ind, vals in proc_df.iterrows():
                fname = "{} Cyl {}-{}.jpg".format(equip, cyl_num, ind.strftime(
                    "%Y%m%d-%H%M%S"))
                plot_cfp(vals.values[0], fname)
        tag_data.append(proc_df)
    merged_df = pd.concat(tag_data, axis=1)
    # Apply the cosine similarity function. If the axis is 1, the function will
    # be applied per row
    ret_df = merged_df.apply(cos_sim, axis=1)
    return ret_df, proc_df


def process_cfp_curve(data, full_tag, cfp_size):
    proc_df = []
    # Identify all places in the data where a CFP curve begins.
    start_CFPs = data[data[full_tag] == CFP_START_VAL].index
    # For the start of each CFP curve, we already know how large the CFP
    # array should be (cfp_size passed parameter), so we can take exactly
    # that many data points after the start of each CFP curve.
    for start_index in start_CFPs:
        at_least = data[data.index > start_index]
        cfp_curve = at_least.ix[0:cfp_size, full_tag]
        cfp_curve[cfp_curve < 0] = 0
        cfp_curve[cfp_curve > 2000] = 0
        proc_df.append([cfp_curve.values])
    label = full_tag.split('.')[-1]
    ret_df = pd.DataFrame(data=proc_df, index=start_CFPs, columns=[label])
    return ret_df


def cos_sim(x):
    avg_x = np.mean(x).reshape(-1, 1)
    res = []
    for i in x:
        col_x = i.reshape(-1, 1)
        res.append(cosine_similarity(col_x, avg_x))
    return res


def plot_cfp(data, fname):
    if len(data) > 0:
        fig = plt.figure(figsize=FSIZE)
        plt.plot(range(0, len(data)), data)
        # Cleanup, labels, and saving the plot
        fig.savefig(fname)
        plt.close()


def plot_data(ship, equip, data):
    if not data.empty:
        fig = plt.figure(figsize=FSIZE)
        # Make a scatter plot of the data, and a threshold based on warn_threshold
        ax = plt.gca()
        data.plot(style='*', ax=ax).legend(loc='center left',
                                           bbox_to_anchor=(1, 0.5),
                                           fancybox=True, shadow=True)
        # Cleanup, labels, and saving the plot
        fig.savefig('{} {}.jpg'.format(ship, equip))
        plt.close()


# MAIN
if __name__ == "__main__":
    user_args = sys.argv
    # ship = str(user_args[1])
    # start_date = str(user_args[2])
    # end_date = str(user_args[3])
    ship = "CAL"
    start_date = "01/01/17"
    end_date = "02/01/17"
    # FIND PARAMETERS
    params = PARAM_DICT[ship]
    cfp_size = CFP_SIZE[ship]
    cfp_service = ship + SERVICE
    # RUN DATA
    cos_data, proc_data = {}, {}
    for param_list in params:
        equip, num_cyl, pres_tag = param_list
        cos_data[equip], proc_data[equip] = run_data(equip, num_cyl, pres_tag,
                                                     cfp_size,
                                                     cfp_service, start_date,
                                                     end_date)
    # PLOT DATA
    for param_list in params:
        equip, num_cyl, pres_tag = param_list
        plot_data(ship, equip, cos_data[equip])