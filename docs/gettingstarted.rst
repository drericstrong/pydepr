====================
 2. Getting Started
====================

Installation
--------------
If Python is already installed on your computer, PyDePr can be installed using 
PyPI by opening a command window and typing:

**pip install pydepr**

Upgrading to a new version of PyDePr can be accomplished by:

**pip install pydepr --upgrade**

The source code of PyDePr is hosted on GitHub at:

https://github.com/drericstrong/pydepr

Python Requirements
--------------------
**Required modules:** matplotlib, numpy, pandas, seaborn, scipy, scikit-learn, sympy, statsmodels

PyDePr is integrated tightly with PyeDNA, with static helper functions for 
pulling data directly from eDNA. PyeDNA is not required for most functions
in PyDePr, but it will be loaded if necessary.

A requirements.txt document is located in the GitHub repository, and all 
package requirements can be installed using the following line in a
command window:

**pip install -r requirements.txt**

Python Version Support
------------------------
Currently, PyDePr only supports Python 3.2+ and is not compatible with
Python 2. If Python 2 support is important to you, please make a pull 
request at:

https://github.com/drericstrong/pydepr

The package maintainer welcomes collaboration.

Importing PyDePr
-----------------
Modules in PyDePr are usually imported into a script using the following lines:

**import pydepr.regression as regr**

**import pydepr.inference as infer**
