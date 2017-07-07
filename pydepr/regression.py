# -*- coding: utf-8 -*-
"""
    pydepr.regression
    --------------------
    PyDePr is a set of tools for processing machine learning models for 
    inferring equipment degradation. This module is meant to preprocess and
    develop regression models.
    
    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

import pandas as pd
import seaborn as sns
from scipy import stats
from sympy import Symbol, Float
import matplotlib.pyplot as plt
import sklearn.linear_model as sklm
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score, mean_absolute_error


class RegressionCurve:
    def __init__(self, model_type="ridge"):
        """
        Initializes a regression curve. The type of regression model to be 
        used is selected here; the parameters of the regression model will
        be selected using cross-validation.
        
        :param model_type: Choose between several types of regression models,
            including: Lasso, ElasticNet, Ridge, LassoLars
        """
        # Initialize the regression model with Nones
        self.model_type = str(model_type).lower()
        self.x_data = None
        self.y_data = None
        self.filters = None
        self.filter_data = None
        self.model = None
        self.y_predict = None
        self.y_residuals = None
        self.y_corrected = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def add_input(self, data, merge_type='outer'):
        """
        Adds values to the X data. If x data already exists, it will merge
        the new data with the existing data using an outer join, by default.
        
        :param data: A pandas DataFrame of input data.
        :param merge_type: If data already exists, it will be merged with
            an 'inner' or 'outer' join. Refers to pandas.concat
            for more information about this behavior.
        """
        df = pd.DataFrame(data)
        # If x_data has not yet been defined yet, then it does not need to be
        # combined with the other data
        if self.x_data is None:
            self.x_data = df
        # Otherwise, merge it using the merge_type method
        else:
            self.x_data = pd.concat([self.x_data, df], axis=1, join=merge_type, 
                                    verify_integrity=True).dropna()
    
    def add_input_edna(self, tag, start_date, end_date, desc_as_label=True,
                       custom_label=None, merge_type='outer'):
        """
        Adds values to the X data. If x data already exists, it will merge
        the new data with the existing data using an outer join, by default.
        
        This helper function will pull data from eDNA, to be used as inputs
        to the RegressionCurve. It is strongly recommended that you use the 
        same start_date and end_date for all data, and that you ensure that
        data actually exists during the time period of interest.
        
        :param tag: The full Site.Service.Tag eDNA tagname
        :param start_date: must be in format mm/dd/yy hh:mm:ss
        :param end_date: must be in format mm/dd/yy hh:mm:ss 
        :param desc_as_label: If true, use the eDNA description as the label
            of the variable in the pandas DataFrame
        :param custom_label: Supply a custom variable label, as a string
        :param merge_type: If data already exists, it will be merged with
            an 'inner' or 'outer' join. Refers to pandas.concat
            for more information about this behavior.
        """
        # Pull the data from eDNA
        data = self._pull_edna(tag, start_date, end_date, desc_as_label,
                               custom_label)
        # DRY- just call the other add_input function
        self.add_input(data, merge_type)
  
    def add_output(self, data):
        """
        Adds values to the Y data. WARNING- if y data already exists, it will
        be overwritten by the new data. Only one Y variable is supported,
        currently.
        
        :param data: A pandas DataFrame of output data.      
        """
        df = pd.DataFrame(data)
        # Warning- currently, only one output can be defined, so adding an
        # output will overwrite the previous one, if it exists
        self.y_data = df
            
    def add_output_edna(self, tag, start_date, end_date, desc_as_label=True,
                       custom_label=None):
        """
        Adds values to the Y data. WARNING- if y data already exists, it will
        be overwritten by the new data. Only one Y variable is supported,
        currently.
        
        This helper function will pull data from eDNA, to be used as outputs
        to the RegressionCurve. It is strongly recommended that you use the 
        same start_date and end_date for all data, and that you ensure that
        data actually exists during the time period of interest.
        
        :param tag: The full Site.Service.Tag eDNA tagname
        :param start_date: must be in format mm/dd/yy hh:mm:ss
        :param end_date: must be in format mm/dd/yy hh:mm:ss 
        :param desc_as_label: If true, use the eDNA description as the label
            of the variable in the pandas DataFrame
        :param custom_label: Supply a custom variable label, as a string     
        """
        # Warning- currently, only one output can be defined, so adding an
        # output will overwrite the previous one, if it exists
        df = self._pull_edna(tag, start_date, end_date, desc_as_label,
                             custom_label) 
        self.y_data = df
    
    def add_filter(self, data, low, high, merge_type='outer'):
        """
        Adds data filters. If a filter already exists, it will merge the new 
        data with the existing data using an outer join, by default.
        
        :param data: A pandas DataFrame of filter data.
        :param low: Values below the "low" parameter will be filtered out
        :param high: Values above the "high" parameter will be filtered out
        :param merge_type: If data already exists, it will be merged with
            an 'inner' or 'outer' join. Refers to pandas.concat
            for more information about this behavior.
        """
        df = pd.DataFrame(data)
        filter_name = df.columns.values[0]    
        # If filter_data has not yet been defined yet, then it does not need 
        # to be combined with the other data
        if self.filter_data is None:
            self.filter_data = df     
            # The "low" and "high" values are stored in a dictionary with the
            # key being the pandas DataFrame column name
            self.filters = {filter_name: [low, high]}
        # Otherwise, merge it using the merge_type method
        else:
            self.filter_data = pd.concat([self.filter_data, df], axis=1, 
                                         join=merge_type, 
                                         verify_integrity=True).dropna()
            self.filters[filter_name] = [low, high]

    def add_filter_edna(self, tag, start_date, end_date, low, high, 
                        desc_as_label=True, custom_label=None, 
                        merge_type='outer'):
        """
        Adds data filters. If a filter already exists, it will merge the new
        data with the existing data using an outer join, by default.
        
        This helper function will pull data from eDNA, to be used as filters
        to the RegressionCurve. It is strongly recommended that you use the 
        same start_date and end_date for all data, and that you ensure that
        data actually exists during the time period of interest.
        
        :param tag: The full Site.Service.Tag eDNA tagname
        :param start_date: must be in format mm/dd/yy hh:mm:ss
        :param end_date: must be in format mm/dd/yy hh:mm:ss 
        :param low: Values below the "low" parameter will be filtered out
        :param high: Values above the "high" parameter will be filtered out
        :param desc_as_label: If true, use the eDNA description as the label
            of the variable in the pandas DataFrame
        :param custom_label: Supply a custom variable label, as a string
        :param merge_type: If data already exists, it will be merged with
            an 'inner' or 'outer' join. Refers to pandas.concat
            for more information about this behavior.
        """
        # Pull the data from eDNA
        data = self._pull_edna(tag, start_date, end_date, desc_as_label,
                               custom_label)
        # DRY- just call the other add_input function
        self.add_filter(data, low, high, merge_type)

    def run(self, model_type=None):
        """
        This method will run the performance curve analysis for the
        initialized data.
        
        :param model_type: Overwrite the regression model chosen during
            initialization. Choices include Lasso, ElasticNet, Ridge, LassoLars
        """
        # Ensure that at least one x and y variable was defined
        if (self.x_data is None) | (self.y_data is None):
            raise Exception("ERROR- specify the x and y data using add_input" +
                            " and add_output")
        # Filter the data using the defined filters    
        xdata, ydata = self._filter_data()
        # Then, divide the data into testing and training sets. This is a
        # necessary step for proper cross-validation of results
        self._divide_train_test(xdata, ydata)
        # Select the correct regression model
        if model_type is not None:
            self.model_type = str(model_type).lower()
        if model_type == "ridge":
            self.model = sklm.RidgeCV(normalize=True)
        if model_type == "lasso":
            self.model = sklm.LassoCV(normalize=True)
        if model_type == "elasticnet":
            self.model = sklm.ElasticNetCV()
        if model_type == "lassolars":
            self.model = sklm.LassoLarsCV(normalize=True)
        else:
            self.model = sklm.RidgeCV(normalize=True)
        # Runs the regression model, based on the training and testing data.
        # Again, why I am storing all these variables within the
        # RegressionCurve object? Because I will need to refer to them when
        # the plots are constructed later, and I don't want to recalculate.
        self.model.fit(self.x_train, self.y_train)
        self.y_predict = self.model.predict(self.x_test)
        self.y_residuals = self.y_test - self.y_predict
        # If more than 1 "x" variable was supplied, we can calculate the
        # "corrected" y by subtracting out the secondary "x" variables, under
        # the assumption that the first "x" variable was the primary one
        self.y_corrected = self.y_test.copy()
        if len(self.model.coef_) > 1:
            for ii, coef in enumerate(self.model.coef_[1:]):
                self.y_corrected -= coef * self.x_test[self.x_labels[ii + 1]]

    def build_equation(self):
        """
        Builds an equation and corrected equation based on the results from
        the constructed model. WARNING- "run" must be called first.

        :return: A tuple containing the equation and the y-corrected equation.
        """
        # Check that the model has actually been run first
        if self.model is None:
            raise Exception("ERROR- 'run' must be called first")
        # For naming convenience of variables, grab several properties which
        # are stored in the object
        x_labels = self.x_data.columns.values
        y_label = self.y_data.columns.values[0]
        intercept = self.model.intercept_[0]
        coefficients = self.model.coef_[0]
        # Each performance curve must contain a "y" variable and at least one
        # "x" variable, so this code is safe
        y = Symbol(y_label)
        x_primary = Symbol(x_labels[0])
        eq_primary = Float(intercept) + Float(coefficients[0]) * x_primary
        x_secondary = []
        eq_secondary = 0
        # If more than 1 "x" variable was supplied, we can calculate the
        # "secondary" equation, which corrects the "y" variable
        if len(coefficients) > 1:
            for ii, coef in enumerate(coefficients[1:]):
                cur_label = x_labels[ii+1].replace(' ', '')
                new_symbol = Symbol(cur_label)
                x_secondary.append(new_symbol)
                eq_secondary += Float(coef) * x_secondary[-1]
        # Find the resulting equation and corrected equation
        equation = 'Value={}'.format(str(eq_primary + eq_secondary))
        corrected_equation = 'Value={}'.format(str(y - eq_secondary))
        return equation, corrected_equation

    def calculate_metrics(self, warn=3, alarm=4):
        """
        Calculates performance metrics for the performance curve.
        Warning- "run" must be called first.

        :param warn: # of standard deviations for the warning limit
        :param alarm: # of standard deviations for the alarm limit
        :return: an array of: [R^2, MAE, EV, Warn limit, Alarm limit]
        """
        if self.model is None:
            raise Exception("ERROR- 'run' must be called first")

        # Calculate each of the metrics
        std = self.y_residuals.std().values[0]
        r2 = r2_score(self.y_test, self.y_predict)
        mae = mean_absolute_error(self.y_test, self.y_predict)
        ev = explained_variance_score(self.y_test, self.y_predict)
        wl = warn * std
        al = alarm * std
        return [r2, mae, ev, wl, al]

    def plot_validation(self, warn=3, alarm=4, title=None, save_fig=False,
                        fig_size=(20, 15)):
        """
        Creates a multi-plot to be used for model validation. WARNIGN- "run" 
        must be called first.
        
        Plot descriptions:
        1. Residuals vs. Time
        2. Residuals vs. Primary Explanatory Factor
        3. Y vs. Yhat Plot
        4. Histogram of the Residuals
        5. Actual Y vs. X
        6. Corrected Y vs. X
        7. Histogram of Actual Y
        8. Histogram of Corrected Y

        :param warn: # of standard deviations for the warning limit
        :param alarm: # of standard deviations for the alarm limit
        :param title: an optional title for the plot
        :param save_fig: if True, the figure will be saved to a file with a
            filename the same as the title
        :param fig_size: the size of the plot
        :return: either a figure plotted in the console, or a figure that is
            saved to a file
        """
        if self.model is None:
            raise Exception("ERROR- 'run' must be called first")
        if title is None:
            title = "Validation Plot"
        metrics = self.calculate_metrics(warn, alarm)
        return self._build_plots(title, metrics, save_fig, fig_size)
    
    # Helper functions that may be called from the functions above    
    def _pull_edna(self, tag, start_date, end_date, desc_as_label=True,
                   custom_label=None):
        # Only import pyedna if the user tries to pull data from eDNA
        import pyedna.ezdna as dna
        return dna.GetHist(tag, start_date, end_date, label=custom_label,
                           desc_as_label=desc_as_label)
        
    def _divide_train_test(self, xdata, ydata):
        # This method will divide the data into training and testing sets.
        x_names = xdata.columns.values
        y_names = ydata.columns.values
        all_data = pd.concat([ydata, xdata], axis=1,
                             verify_integrity=True).dropna()
        train, test = train_test_split(all_data)
        # Store the training and testing data in the object itself, because
        # we will need to refer to the training and testing data when we
        # are constructing the plots
        self.x_train = train[x_names]
        self.y_train = train[y_names]
        self.x_test = test[x_names]
        self.y_test = test[y_names]
        
    def _filter_data(self):
        # Concatenate all the data together, since we are going to be filtering
        # both the x and y data with the same filter
        x_names = self.x_data.columns.values
        y_names = self.y_data.columns.values
        all_data = pd.concat([self.y_data, self.x_data], axis=1,
                             verify_integrity=True).dropna()
        # key is the variable name of the filter
        # value[0] is the low limit, and value[1] is the high limit
        for key, value in self.filters.items():
            all_data = all_data[(all_data[key] >= value[0]) &
                                (all_data[key] <= value[1])]
        return all_data[x_names], all_data[y_names]

    def _build_plots(self, title, metrics, save_fig, fig_size):
        # Unpack the statistical metrics and labels, so they will be slightly 
        # easier to refer to in the following code
        r2, mae, ev, wl, al = metrics
        x_labels = self.x_data.columns.values
        y_label = self.y_data.columns.values[0]
        # Create a matplotlib figure with 8 subplots. This figure is intended
        # to be used to verify the results of the regression model
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4,
            figsize=fig_size)
        # Build the equation for the regression model, since it will be used 
        # in the title of the plot
        equation = self.build_equation()
        f.suptitle(title + "\n" + equation[0], fontsize=16)
        # This code will build the 8 subplots of the figure. Descriptions:
        # 1. Residuals vs. Time
        # 2. Residuals vs. Primary Explanatory Factor
        # 3. Y vs. Yhat Plot
        # 4. Histogram of the Residuals
        # 5. Actual Y vs. X
        # 6. Corrected Y vs. X
        # 7. Histogram of Actual Y
        # 8. Histogram of Corrected Y
        prim_data = self.x_test.iloc[:, 0].values
        self._build_residuals(ax1, range(len(self.y_residuals)),
                              'Time (Index)', wl, al)
        self._build_residuals(ax2, prim_data, x_labels[0], wl, al)
        self._build_yyhat(ax3, mae, r2, y_label)
        self._build_histogram(ax4, self.y_residuals, 'Residuals')
        self._build_scatter(ax5, prim_data, self.y_test, x_labels[0],
                            y_label, 'Actual')
        self._build_scatter(ax6, prim_data, self.y_corrected, x_labels[0], 
                            y_label, 'Corrected')
        self._build_histogram(ax7, self.y_test, 'Actual {}'.format(y_label))
        self._build_histogram(ax8, self.y_corrected,
                              'Corrected {}'.format(y_label))
        # Save the figure, if the user wants
        if save_fig:
            plt.savefig('{}.jpg'.format(title))
            plt.close()

    def _build_residuals(self, cur_ax, x_values, x_label, wl, al):
        # Build a scatter plot of the residuals against one of the values that
        # was supplied in the figure
        # cur_ax is the matplotlib axes upon which to draw this plot
        # x_values are the data points to place on the x axis (the y axis is
        #         constained to be the residuals)
        # x_label is a label for the x axis
        # wl is the +/- warning limit
        # al is the +/- alarm limit
        cur_ax.scatter(x_values, self.y_residuals)
        cur_ax.set_title('Residuals vs. {}'.format(x_label))
        cur_ax.set_ylabel('Residual Value')
        cur_ax.set_xlabel(x_label)
        self._plot_limits(cur_ax, min(x_values), max(x_values), wl, al)

    def _build_yyhat(self, cur_ax, mae, r2, y_label):
        # A Y-Yhat plot shows the predictions against the actual values
        # cur_ax is the matplotlib axes upon which to draw this plot
        # mae is the mean absolute error, from self.calculate_metrics
        # r2 is the R^2 value, from self.calculate_metrics  
        yt = self.y_test.values
        yp = self.y_predict
        abs_min = min([yt.min(), yp.min()])
        abs_max = max([yt.max(), yp.max()])
        cur_ax.scatter(yt, yp)
        # Creates a text box that shows the mean absolute error and R^2
        self._plot_text_box(cur_ax, '$MAE=%.3f$\n$R2=%.3f$' % (mae, r2))
        title = 'Predicted {} vs. Actual {}'.format(y_label, y_label)
        cur_ax.set_title(title)
        cur_ax.axis([abs_min, abs_max, abs_min, abs_max])
        cur_ax.plot([abs_min, abs_max], [abs_min, abs_max], c="k")
        cur_ax.set_ylabel('Predicted {}'.format(y_label))
        cur_ax.set_xlabel('Actual {}'.format(y_label))

    @staticmethod
    def _build_scatter(cur_ax, x, y, x_label, y_label, type_label):
        # Builds a typical scatter plot using the x and y data
        # cur_ax is the matplotlib axes upon which to draw this plot
        # x and y are the data to place on the x and y axes, respectively
        # x_label and y_label are labels for the axes
        # type_label is meant to be either 'actual' (plot 5) or 'corrected'
        #       (plot 6), depending on the type of plot being shown. 
        cur_ax.scatter(x, y)
        cur_ax.set_title('{} {} vs. {}'.format(type_label, y_label, x_label))
        cur_ax.set_ylabel('{} {}'.format(type_label, y_label))
        cur_ax.set_xlabel(x_label)

    def _build_histogram(self, cur_ax, values, label):
        # Builds a histogram of the values that were supplied
        # cur_ax is the matplotlib axes upon which to draw this plot
        # values are the data points for the histogram
        # label is either 'actual' (plot 7) or 'corrected' (plot 8), 
        #       depending on the type of plot
        sns.distplot(values, ax=cur_ax, kde=False, fit=stats.norm)
        # Creates a text box that shows the mean, median, and std
        string = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % \
                 (values.mean(), values.median(), values.std())
        self._plot_text_box(cur_ax, string)
        cur_ax.set_title('Histogram of {}'.format(label))
        cur_ax.set_ylabel('# Occurrences')
        cur_ax.set_xlabel('{} Value'.format(label))
   
    @staticmethod
    def _plot_limits(cur_ax, min_val, max_val, warn, alarm):
        # Additional helper function to plot upper and lower warning and alarm 
        # limits on the axis specified by cur_ax
        # cur_ax is the matplotlib axes upon which to draw the limits
        # min_val and max_val are used so that the line will span the full
        #        range of the data, no more, no less
        # warn is the +/- warning limit, plotted with a yellow line
        # alarm is the +/- alarm limit, plotted with a red line
        cur_ax.plot([min_val, max_val], [0, 0], c='g')
        cur_ax.plot([min_val, max_val], [warn, warn], c='y')
        cur_ax.plot([min_val, max_val], [-warn, -warn], c='y')
        cur_ax.plot([min_val, max_val], [alarm, alarm], c='r')
        cur_ax.plot([min_val, max_val], [-alarm, -alarm], c='r')

    @staticmethod
    def _plot_text_box(cur_ax, text):
        # Additional helper function to plot a text box on an axes
        # cur_ax is the matplotlib axes upon which to draw the text box
        # text denotes the text to be placed in the box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        cur_ax.text(0.05, 0.95, text, transform=cur_ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)