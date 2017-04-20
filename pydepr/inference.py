# -*- coding: utf-8 -*-
"""
    pydepr.inference
    ------------------
    PyDePr is a set of tools for processing degradation models. This module
    contains tools for inferring degradation based on evidence.

    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

import math
import numpy as np
import skfuzzy as fuzz


class FuzzyStates:
    def __init__(self, normal, warning, alarm, danger, steps=10000):
        """
        FuzzyStates uses a fuzzy linear interpolation between the four user-
        defined states. Warning- the supplied values must be the "midpoints",
        or the thresholds at which the fuzzy membership for the state should
        be equal to 100%.

        Total membership across all four states must always be equal to 100%.

        :param normal: signal value under normal operating conditions.
        :param warning: signal value that begins to indicate an anomaly.
        :param alarm: onset of significant degradation.
        :param danger: signal indicates that component can no longer meet
            design guidelines.
        :param steps: an optional parameter for the # of steps of the
            range of fuzzy values. Must be at least 100.
        """
        # Steps must be at least 10
        if steps < 100:
            steps = 100
        # The states must be defined as floats.
        self.normal = float(normal)
        self.warning = float(warning)
        self.alarm = float(alarm)
        self.danger = float(danger)
        # Whether the direction of the states is increasing or decreasing
        self.direction_increasing = normal < danger
        # The accuracy of the output depends on the step size selected, so
        # we will round based on the log
        self.round = int(math.log(steps, 10) - 1)

        # Define the range of possible values. Since we don't know if the
        # values are in increasing or decreasing order, we can take the min
        # and max of both the normal and danger to find the range
        start = min(normal, danger)
        end = max(normal, danger)
        span = abs(start - end)
        step_size = span / steps
        self.x = np.arange(start, end, step_size)

        # Scikit-fuzzy requires that the states be defined in increasing order
        # so we have to make adjustments depending on which way the user
        # supplied the states to this function.
        if self.direction_increasing:
            self.x_norm = fuzz.trimf(self.x, [normal, normal, warning])
            self.x_warn = fuzz.trimf(self.x, [normal, warning, alarm])
            self.x_alarm = fuzz.trimf(self.x, [warning, alarm, danger])
            self.x_dang = fuzz.trimf(self.x, [alarm, danger, danger])
        else:
            self.x_norm = fuzz.trimf(self.x, [warning, normal, normal])
            self.x_warn = fuzz.trimf(self.x, [alarm, warning, normal])
            self.x_alarm = fuzz.trimf(self.x, [danger, alarm, warning])
            self.x_dang = fuzz.trimf(self.x, [danger, danger, alarm])

    def calc_memberships(self, value):
        """
        Calculates the memberships of the four states based on the user-
        supplied value.

        :param value: the value of the signal to be evaluated.
        :return: An array with four values: [normal_membership,
            warning_membership, alarm_membership, danger_membership]
        """
        # Check if the value is outside the bounds. This depends on whether
        # the values are increasing or decreasing, as well.
        if (self.direction_increasing & (value <= self.normal)) or \
           (~self.direction_increasing & (value >= self.normal)):
                return [1.0, 0, 0, 0]
        if (self.direction_increasing & (value >= self.danger)) or \
           (~self.direction_increasing & (value <= self.danger)):
                return [0, 0, 0, 1.0]

        # If within bounds, calculate the membership as a probability
        # in each state
        memberships = []
        for state in [self.x_norm, self.x_warn, self.x_alarm, self.x_dang]:
            state_membership = fuzz.interp_membership(self.x, state, value)
            round_membership = round(state_membership, self.round)
            memberships.append(round_membership)
        return memberships


class Evidence:
    def __init__(self, thresholds, cpt, name=None, default_state=0):
        """
        This class defines an Evidence input to a FailureMode. Any
        symptom which indicates degradation within a component may be used
        as an Evidence input.

        :param thresholds: A list of thresholds in order: [normal, warning,
            alarm, danger, extreme danger]
        :param cpt: the conditional probability table for the evidence, which
            specifies the marginal probability distribution of the truth of
            the failure mode, depending on selected state. Must be an array
            of length 4. For example: [0.1, 0.2, 0.3, 0.4]
        :param name: an optional description or identifier.
        :param default_state: the default state of the evidence if bad input
            is received. 0 defaults to the prior, 1 defaults to Normal,
            2 defaults to Warning, 3 defaults to Alarm, and 4 defaults to
            Danger. Recommended values are either 0 or 1.
        """
        # Error checking to ensure the user-supplied parameters are okay.
        if len(cpt) != 4:
            raise AssertionError('Length of the cpt must be 4')
        if len(thresholds) != 5:
            raise AssertionError('Length of the thresholds must be 5')
        if default_state < 0 | default_state > 4:
            raise AssertionError('default_state must be between 0 and 4')
        self.thresholds = thresholds
        self.cpt = cpt
        self.name = name
        self.default_state = default_state

        # Find the midpoints between the 5 thresholds, which specify the
        # four states. The four states will be translated into fuzzy logic.
        midpoints = [((thresholds[ii] + thresholds[ii+1]) / 2)
                     for ii in range(4)]
        self.states = FuzzyStates(*midpoints)

    def calc_bi_given_a(self, value):
        """
        Calculates the probability of B_i given A for the evidence node.

        :param value: the current value of the evidence.
        :return: P(B_i|A)
        """
        ms = self.states.calc_memberships(value)
        p_bi_a = sum(m*p for m, p in zip(ms, self.cpt))
        return p_bi_a

    def calc_bi_given_not_a(self, value):
        """
        Calculates the probability of B_i given NOT A for the evidence node.

        :param value: the current value of the evidence.
        :return: P(B_i|~A)
        """
        ms = self.states.calc_memberships(value)
        # ::-i will reverse the CPT matrix
        p_bi_not_a = sum(m * p for m, p in zip(ms, self.cpt[::-1]))
        return p_bi_not_a


class ContraryEvidence:
    def __init__(self, thresholds, priors, name=None):
        """
        This class defines a contrary evidence input to a FailureMode. Any
        symptom which will excuplate the occurrence of the failure mode can be
        used as a ContraryEvidence node.

        :param thresholds: A list of 4 thresholds in order: [benign, uncertain,
            conflicting, incompatible]
        :param priors: an array of updated priors for the failure mode,
            based on the selected ContraryEvidence state. Must be length 4.
        :param name: an optional description or identifier.
        """
        # Error checking to ensure the user-supplied parameters are okay.
        if len(priors) != 4:
            raise AssertionError('Length of the priors must be 4')
        if len(thresholds) != 4:
            raise AssertionError('Length of the thresholds must be 4')

        # Calculate the midpoints of the thresholds to find the states
        self.states = thresholds
        self.priors = priors
        self.name = name
        self.direction_increasing = thresholds[0] < thresholds[3]

    def calculate_prior(self, value):
        """
        Calculates an updated prior for the FailureMode based on the supplied
        value.

        :param value: current value of the contrary evidence signal.
        :return: updated prior for the failure mode.
        """
        # Iterate through the states and check if the value is greater than the
        # state. If so, return that prior.
        for state, prior in zip(self.states[1:], self.priors[:-1]):
            # If the states are increasing, we want to start with the "benign"
            # states, but if the states are decreasing, we start at other end
            if (self.direction_increasing & (value <= state)) | \
               (~self.direction_increasing & (value >= state)):
                return prior
        return self.priors[-1]


class Conclusion:
    def __init__(self, prior, name=None, evidence=None, contrary=None):
        """
        Defines a conclusion in terms of evidence and contrary evidence.

        :param name: an optional description or identifier
        :param prior: the prior probability of occurrence
        :param evidence: an array of "Evidence" class nodes which define
            symptoms associated with the conclusion. At least 1 must exist.
        :param contrary: a single "ContraryEvidence" class node which
            contradicts the conclusion. A maximum of 1 may exist.
        """
        # Basic error checking of the user inputs
        if not evidence:
            raise AssertionError("At least one evidence must exist.")
        if type(evidence) is not list:
            raise AssertionError("Evidence must be in a list.")
        if type(contrary) is list:
            raise AssertionError("Contrary evidence cannot be a list.")
        self.name = name
        self.prior = prior
        self.evidence = evidence
        self.contrary = contrary

    def _prob_b_given_a(self, values):
        # Calculates the probability of b given a, which is a series of
        # multiplications of all the bi given a of all evidence nodes
        p_b_a = 1
        for ii, evidence in enumerate(self.evidence):
            p_b_a *= evidence.calc_bi_given_a(values[ii])
        return round(p_b_a, 5)

    def _prob_b_given_not_a(self, values):
        # Calculates the probability of b given not a, which is a series of
        # multiplications of all the bi given not a of all evidence nodes
        p_b_not_a = 1
        for ii, evidence in enumerate(self.evidence):
            p_b_not_a *= evidence.calc_bi_given_not_a(values[ii])
        return round(p_b_not_a, 5)

    def _prob_b(self, values, prior=None):
        # Calculates the probability of b, which is p(a)*_prob_b_given_a +
        # p(~a)*_prob_b_given_not_a
        # If a prior is supplied, calculate assuming that prior, else
        # calculate assuming the original prior
        if not prior:
            prior = self.prior
        p_b = (self._prob_b_given_a(values) * prior) + \
              (self._prob_b_given_not_a(values) * (1 - prior))
        return round(p_b, 5)

    def calc_prob_a_given_b(self, values, contrary=None):
        """
        Given a list of values, calculates the associated probability
        that the conclusion is occurring.

        :param values: An array that must be the same size as the number of
            evidence nodes.
        :param contrary: supply optional contrary evidence, if a contrary
            evidence node was defined.
        :return: probability of a given b (the probability that the failure
            mode is occurring).
        """
        # User-supplied parameters error checking
        if type(values) is not list:
            raise AssertionError("Values must be a list.")
        if len(values) != len(self.evidence):
            er = "Values must be the same length as the number of evidences."
            raise AssertionError(er)

        # If contrary evidence exists, we want to update the prior. Else,
        # the prior should default to the original value.
        prior = self.prior
        if self.contrary:
            prior = self.contrary.calculate_prior(contrary)

        # Calculate the result using the inner functions above
        prob_a_given_b = (self._prob_b_given_a(values) * prior) / (
                          self._prob_b(values, prior))
        return round(prob_a_given_b, 5)
