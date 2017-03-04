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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class PressureCurve():
    def __init__(self, data):
        """
        Defines a pressure curve, with functions allowing calculation of
        various degradation features.

        :param data:
        """
        self.data = data

    @staticmethod
    def list_from_edna(tag, start_date, end_date, start_val=-7291290,
                       array_size=7200):
        """
        Returns a list of pressure curves from eDNA.

        :param tag: the eDNA tag to pull the pressure curve
        :param start_date: the beginning of the data pull
        :param end_date: the end of the data pull
        :param start_val: the value in history that defines the start of the
            array
        :param array_size: the expected size of the pressure array
        :return: a pandas DataFrame, with each row as a single pressure array
        """
        import pyedna.ezdna as dna
        # Pull the data and detect where the beginning of each curve lies
        df = dna.GetHistRaw(tag, start_date, end_date, high_speed=True)
        start_indices = df[df.tag == start_val].index
        proc_df = []

        # For the start of each  curve, we already know how large the
        # array should be (cfp_size passed parameter), so we can take exactly
        # that many data points after the start of each CFP curve.
        for start_index in start_indices:
            end_index = start_index + 120
            curve = df[(df.index > start_index) & (df.index < end_index)]
            curve = curve.resample("1MS")
            proc_df.append([curve.values])
        label = tag.split('.')[-1]
        ret_df = pd.DataFrame(data=proc_df, index=start_index, columns=[label])
