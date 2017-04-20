====================
 3. Regression
====================
PyDePr supports the construction of regression models, with built-in
visual model validation. The following features are available:

* Initialize data from either a pandas DataFrame or eDNA
* Automatically perform Ridge Regression
* Calculate model performance metrics
* Build the model equation in a user-friendly form
* Create a series of plots for model validation and visualization

All code in this section assumes that you have imported PyDePr like:

**import pydepr.regression as regr**

Regression Curve
------------------
The base class in this namespace is the RegressionCurve class. You can
initialize a RegressionCurve by supplying it with x and y data:

**curve = regr.RegressionCurve(y_data, x_data)**

Next, the "run" function will automatically create a Ridge Regression
model using the x and y data:

**curve.run()**

Once the regression curve has been run, the model validation metrics
can be found (next section).

Model Validation
------------------
PyDePr will generate a plot which can be used for validation of the 
regression model, using the following function:

**f = curve.plot_validation()**

Warning and alarm limits (based on standard deviations) can also be
supplied to the plots:

**f = curve.plot_validation(warn=2, alarm=3)**

The validation metrics generated for the above plot can also be 
found directly:

**metrics = curve.calculate_metrics()**

Equation Building
-------------------
The RegressionCurve class will automatically build equations based 
on the results from the Ridge Regression:

**eq, corr_eq = curve.build_equation()**

The first value returned will be the full regression model equation,
while the second value returned will be the "corrected" equation
(more explanation below).

The regular equation is of the format (for ease of import into eDNA):

**Value = AX + BZ**

The corrected equation is of the format:

**Value = BZ - Y**

Initialization from eDNA
--------------------------
Using PyeDNA, regression models can be built automatically by simply
supplying eDNA tags:

**curve = regr.RegressionCurve.from_edna(y_tag, x_tags, dates=[DATES])**

This is a static function that returns a RegressionCurve.

The following parameters may be set:

* y_tag: an eDNA tag of the form Site.Service.Tag, enclosed by brackets. For example, ["MDSSCSC1.CALCALC.ADE1CA01"]
* x_tags: a list of eDNA tags, enclosed by brackets. For example: ["MDSSCSC1.CALCALC.ADE1CA01", "MDSSCSC1.CALCALC.ADE1CA02", "MDSSCSC1.CALCALC.ADE1CA03"]
* dates: NOT OPTIONAL. An array of arrays, where the innermost array is of the form [start_date, end_date]. This specifies which data to pull. For example: [["01/01/2016", "02/01/2016"], ["04/01/2017","07/01/2016"]]
* y_label: Override the eDNA description with your own label.
* x_labels: Override the eDNA description with your own labels. WARNING- must be exactly the same size as x_tags.