# -*- coding: utf-8 -*-
"""
    pydepr
    ~~~~~~~~~~
    PyDePr is a set of tools for processing degradation models.

    :copyright: (c) 2017 by Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

__version__ = '0.12'
from .regression import PerformanceCurve
from .inference import FuzzyStates, Evidence, ContraryEvidence, FailureMode
