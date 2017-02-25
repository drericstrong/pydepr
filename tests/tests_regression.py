# -*- coding: utf-8 -*-
"""
    pydepr tests_regression
    ~~~~~~~~~~~~~~~~~~~~~~~
    A set of unit tests for the regression module.

    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

import unittest
import pandas as pd
import pydepr.regression as reg


class TestPerformanceCurve(unittest.TestCase):
    def test_fromEdna_noDates(self):
        with self.assertRaises(TypeError):
            pc = reg.PerformanceCurve(pd.DataFrame(), pd.DataFrame())

    def test_fromEdna_badXLabelSize(self):
        with self.assertRaises(IndexError):
            pc = reg.PerformanceCurve(pd.DataFrame(), pd.DataFrame())

    def test_fromEdna_XnotList(self):
        with self.assertRaises(TypeError):
            pc = reg.PerformanceCurve(pd.DataFrame(), pd.DataFrame())

    def test_fromEdna_YnotList(self):
        with self.assertRaises(TypeError):
            pc = reg.PerformanceCurve(pd.DataFrame(), pd.DataFrame())

if __name__ == '__main__':
    unittest.main()
