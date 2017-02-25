# PyDePr
PyDePr (Python Degradation Preprocessing) is a toolkit designed to facilitate
the preprocessing of degradation models (i.e. condition-monitoring models for
components such as engines or generators). Currently, supported modules 
include:

* Regression modeling
* Waveform processing

Additional modules are in development.

## Dependencies
**Required libraries**: matplotlib, numpy, pandas, seaborn, scipy, 
scikit-learn, sympy, statsmodels

**Optional libraries**: pyedna (for pulling data from eDNA)

## Regression Modeling
PyDePr supports the construction of multiple regression models, with built-in
visual model validation. The following features are available:

* Initialize data from either a pandas DataFrame or eDNA
* Automatically perform Ridge Regression
* Calculate model performance metrics
* Build the model equation in a user-friendly form
* Create a series of plots for model validation and visualization

![Regression](/images/Regression.jpg)

## Waveform Processing
Still in development.



