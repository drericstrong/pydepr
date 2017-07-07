=====================
 3. Regression
=====================
PyDePr supports the construction of regression models, with built-in
visual model validation. The following features are available:

* Supply data from either a pandas DataFrame or eDNA
* Automatically perform Ridge Regression, Lasso, LassoLars, and ElasticNet
* Calculate model performance metrics
* Build the model equation in a user-friendly form
* Create a series of plots for model validation and visualization

All code in this section assumes that you have imported PyDePr like:

**import pydepr.regression as regr**

Regression Curve
------------------
The base class in this namespace is the RegressionCurve class. You can
initialize a RegressionCurve by supplying it with a model type:

**curve = regr.RegressionCurve(model_type="ridge")**

Possible values for the model_type parameter are "Lasso", "ElasticNet", 
"Ridge", and "LassoLars". The default value is "Ridge".

Next, provide the model with inputs (x data) and outputs (y data),
using the **add_input** and **add_output** methods, respectively:

**curve.add_input(x_data)**

**curve.add_output(y_data)**

If x data already exists, it will merge the new data with the existing 
data using an outer join, by default. However, only one y variable 
may be specified, so the **add_output** function will overwrite all the 
existing y_data.

Optionally, you can specify data filters using the add_filter method:

**curve.add_filter(filter_data, low_value, high_value)**

The x and y data will be filtered based on when the filter_data is 
greater than or equal to the low_value, and less than or equal to the 
high_value.

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