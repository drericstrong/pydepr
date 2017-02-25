# -*- coding: utf-8 -*-
"""
    pydepr.regression
    ~~~~~~~~~~~~~~~
    PyDePr is a set of tools for processing degradation models. This module
    contains tools for processing and validating regression models.

    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

# TODO- dynamic warning and alarm thresholds

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sympy import Symbol, Float
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score, mean_absolute_error


class PerformanceCurve:
    def __init__(self, y_data, x_data, y_label=None, x_labels=None):
        """
        Runs the performance curve regression using a pandas DataFrame.

        :param y_data: a pandas DataFrame for the y data.
        :param x_data: a pandas DataFrame containing one or more X columns
        :param y_label: Override the eDNA description with your own label.
        :param x_labels: Override the eDNA description with your own labels.
            WARNING- must be exactly the same size as columns in x_data.
        """
        # Check that the correct number of x labels were supplied
        if x_labels:
            if len(x_data.columns.values) != len(x_labels):
                raise Exception('ERROR- x labels must be same size as x tags.')
            self.x_labels = x_labels
        else:
            self.x_labels = x_data.columns.values
        if y_label:
            self.y_label = y_label
        else:
            self.y_label = y_data.columns.values

        # Save the x and y data, and initialize some properties
        self.y_data = y_data
        self.x_data = x_data
        self._init_none()

    def _init_none(self):
        # This function initializes some properties as None
        self.model = None
        self.y_predict = None
        self.y_residuals = None
        self.y_corrected = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @staticmethod
    def from_edna(y_tag, x_tags, dates=None, y_label=None, x_labels=None):
        """
        Runs the performance curve regression by pulling data from eDNA.

        :param y_tag: an eDNA tag of the form Site.Service.Tag, enclosed by
            brackets. For example, ["MDSSCSC1.CALCALC.ADE1CA01"]
        :param x_tags: a list of eDNA tags, enclosed by brackets. For example:
            ["MDSSCSC1.CALCALC.ADE1CA01", "MDSSCSC1.CALCALC.ADE1CA02",
             "MDSSCSC1.CALCALC.ADE1CA03"]
        :param dates: NOT OPTIONAL. An array of arrays, where the innermost
            array is of the form [start_date, end_date]. This specifies which
            data to pull. For example: [["01/01/2016", "02/01/2016"],
            ["04/01/2017","07/01/2016"]]
        :param y_label: Override the eDNA description with your own label.
        :param x_labels: Override the eDNA description with your own labels.
            WARNING- must be exactly the same size as x_tags.
        """
        # Do some basic error checking first.
        if not dates:
            raise TypeError('ERROR- dates are not supplied or in bad format.')
        if x_labels:
            if len(x_tags) != len(x_labels):
                raise IndexError('ERROR- xlabels must be same size as xtags.')
        if type(y_tag) is not list:
            raise TypeError('ERROR- y tag must be enclosed with brackets.')
        if type(x_tags) is not list:
            raise TypeError('ERROR- x tags must be enclosed with brackets.')

        # Pull the description from each tag, to be used as the column label
        import pyedna.ezdna as dna
        if not y_label:
            y_label = dna.GetTagDescription(y_tag[0])
        if not x_labels:
            x_labels = [dna.GetTagDescription(tag) for tag in x_tags]

        # Multiple time periods may be specified in the dates array. This is
        # so the user can give good, representative data instead of a blanket
        # time period.
        x_dfs, y_dfs = [], []
        for start, end in dates:
            y_df = dna.GetMultipleTags(y_tag, start, end, desc_as_label=True)
            x_df = dna.GetMultipleTags(x_tags, start, end, desc_as_label=True)
            y_dfs.append(y_df)
            x_dfs.append(x_df)

        # Concat along rows, since there are multiple time periods
        y_data = pd.concat(y_dfs, verify_integrity=True)
        x_data = pd.concat(x_dfs, verify_integrity=True)
        return PerformanceCurve(y_data, x_data, y_label, x_labels)

    def run(self):
        """
        This method will run the performance curve analysis for the
        initialized data.
        """
        # Create a ridge regression cross-validation model
        self._divide_train_test()
        self.model = RidgeCV(alphas=np.logspace(-3, 3, num=50), normalize=True)
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

    def _divide_train_test(self):
        # This method will divide the data into training and testing sets.
        x_names = self.x_data.columns.values
        y_names = self.y_data.columns.values
        all_data = pd.concat([self.y_data, self.x_data], axis=1,
                             verify_integrity=True).dropna()
        train, test = train_test_split(all_data)
        self.x_train = train[x_names]
        self.y_train = train[y_names]
        self.x_test = test[x_names]
        self.y_test = test[y_names]

    def build_equation(self):
        """
        Builds an equation and corrected equation based on the results from
        the constructed model. Warning- "run" must be called first.

        :return: A tuple containing the equation and the y-corrected equation.
        """
        if self.model is None:
            raise Exception("ERROR- 'run' must be called first")
        intercept = self.model.intercept_[0]
        coefficients = self.model.coef_[0]

        # Each performance curve must contain a "y" variable and at least one
        # "x" variable, so this code is safe
        y = Symbol(self.y_label)
        x_primary = Symbol(self.x_labels[0])
        eq_primary = Float(intercept) + Float(coefficients[0]) * x_primary
        x_secondary = []
        eq_secondary = 0

        # If more than 1 "x" variable was supplied, we can calculate the
        # "secondary" equation, which
        if len(coefficients) > 1:
            for ii, coef in enumerate(coefficients[1:]):
                cur_label = self.x_labels[ii+1].replace(' ', '')
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
        Creates a multi-plot to be used for model validation.
        Warning- "run" must be called first.

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
        self._build_plots(title, metrics, save_fig, fig_size)

    def _build_plots(self, title, metrics, save_fig, fig_size):
        # A "master" function to build all 8 plots
        r2, mae, ev, wl, al = metrics
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4,
            figsize=fig_size)
        equation = self.build_equation()
        f.suptitle(title + "\n" + equation[0], fontsize=16)

        # Each of the 8 plots will be built as follows:
        prim_data = self.x_test.iloc[:, 0].values
        self._build_residuals(ax1, range(len(self.y_residuals)),
                              'Time (Index)', wl, al)
        self._build_residuals(ax2, prim_data, self.x_labels[0], wl, al)
        self._build_yyhat(ax3, mae, r2)
        self._build_histogram(ax4, self.y_residuals, 'Residuals')
        self._build_scatter(ax5, prim_data, self.y_test, self.x_labels[0],
                            self.y_label, 'Actual')
        self._build_scatter(ax6, prim_data, self.y_corrected,
                            self.x_labels[0], self.y_label, 'Corrected')
        self._build_histogram(ax7, self.y_test,
                              'Actual {}'.format(self.y_label))
        self._build_histogram(ax8, self.y_corrected,
                              'Corrected {}'.format(self.y_label))

        # Save the figure, if the user wants
        if save_fig:
            plt.savefig('{}.jpg'.format(title))
            plt.close()

    def _build_residuals(self, cur_ax, x_values, x_label, wl, al):
        cur_ax.scatter(x_values, self.y_residuals)
        cur_ax.set_title('Residuals vs. {}'.format(x_label))
        cur_ax.set_ylabel('Residual Value')
        cur_ax.set_xlabel(x_label)
        self._plot_limits(cur_ax, min(x_values), max(x_values), wl, al)

    def _build_yyhat(self, cur_ax, mae, r2):
        yt = self.y_test.values
        yp = self.y_predict
        abs_min = min([yt.min(), yp.min()])
        abs_max = max([yt.max(), yp.max()])
        cur_ax.scatter(yt, yp)
        self._plot_text_box(cur_ax, '$MAE=%.3f$\n$R2=%.3f$' % (mae, r2))
        title = 'Predicted {} vs. Actual {}'.format(self.y_label, self.y_label)
        cur_ax.set_title(title)
        cur_ax.axis([abs_min, abs_max, abs_min, abs_max])
        cur_ax.plot([abs_min, abs_max], [abs_min, abs_max], c="k")
        cur_ax.set_ylabel('Predicted {}'.format(self.y_label))
        cur_ax.set_xlabel('Actual {}'.format(self.y_label))

    def _build_histogram(self, cur_ax, values, label):
        sns.distplot(values, ax=cur_ax, kde=False, fit=stats.norm)
        string = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % \
                 (values.mean(), values.median(), values.std())
        self._plot_text_box(cur_ax, string)
        cur_ax.set_title('Histogram of {}'.format(label))
        cur_ax.set_ylabel('# Occurrences')
        cur_ax.set_xlabel('{} Value'.format(label))

    @staticmethod
    def _plot_limits(cur_ax, min_val, max_val, warn, alarm):
        # This function will plot upper and lower warning and alarm limits
        # on the axis specified by cur_ax
        cur_ax.plot([min_val, max_val], [0, 0], c='g')
        cur_ax.plot([min_val, max_val], [warn, warn], c='y')
        cur_ax.plot([min_val, max_val], [-warn, -warn], c='y')
        cur_ax.plot([min_val, max_val], [alarm, alarm], c='r')
        cur_ax.plot([min_val, max_val], [-alarm, -alarm], c='r')

    @staticmethod
    def _build_scatter(cur_ax, x, y, x_label, y_label, type_label):
        cur_ax.scatter(x, y)
        cur_ax.set_title('{} {} vs. {}'.format(type_label, y_label, x_label))
        cur_ax.set_ylabel('{} {}'.format(type_label, y_label))
        cur_ax.set_xlabel(x_label)

    @staticmethod
    def _plot_text_box(cur_ax, text):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        cur_ax.text(0.05, 0.95, text, transform=cur_ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
