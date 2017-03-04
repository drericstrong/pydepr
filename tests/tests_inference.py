# -*- coding: utf-8 -*-
"""
    pydepr tests_inference
    ~~~~~~~~~~~~~~~~~~~~~~~
    A set of unit tests for the inference module.

    :copyright: (c) 2017 Eric Strong.
    :license: Refer to LICENSE.txt for more information.
"""

import unittest
import pydepr.inference as infer


class TestFuzzyStates(unittest.TestCase):
    def test_fuzzy_directionDetection(self):
        fs = infer.FuzzyStates(0, 1, 2, 3)
        self.assertTrue(fs.direction_increasing)
        fs = infer.FuzzyStates(3, 2, 1, 0)
        self.assertFalse(fs.direction_increasing)

    def test_fuzzy_badOrdering(self):
        with self.assertRaises(AssertionError):
            fs = infer.FuzzyStates(0, 2, 1, 3)

    def test_fuzzy_lowVal_increasing(self):
        fs = infer.FuzzyStates(0, 1, 2, 3)
        expected_val = [1, 0, 0, 0]
        low_val = fs.calc_memberships(0)
        lowest_val = fs.calc_memberships(-1)
        self.assertEqual(expected_val, low_val)
        self.assertEqual(expected_val, lowest_val)

    def test_fuzzy_highVal_increasing(self):
        fs = infer.FuzzyStates(0, 1, 2, 3)
        expected_val = [0, 0, 0, 1]
        high_val = fs.calc_memberships(3)
        highest_val = fs.calc_memberships(4)
        self.assertEqual(expected_val, high_val)
        self.assertEqual(expected_val, highest_val)

    def test_fuzzy_highVal_decreasing(self):
        fs = infer.FuzzyStates(3, 2, 1, 0)
        expected_val = [1, 0, 0, 0]
        low_val = fs.calc_memberships(3)
        lowest_val = fs.calc_memberships(4)
        self.assertEqual(expected_val, low_val)
        self.assertEqual(expected_val, lowest_val)

    def test_fuzzy_lowVal_decreasing(self):
        fs = infer.FuzzyStates(3, 2, 1, 0)
        expected_val = [0, 0, 0, 1]
        high_val = fs.calc_memberships(0)
        highest_val = fs.calc_memberships(-1)
        self.assertEqual(expected_val, high_val)
        self.assertEqual(expected_val, highest_val)

    def test_fullMembership_increasing(self):
        fs = infer.FuzzyStates(0, 1, 2, 3)
        warning_expected = [0, 1, 0, 0]
        alarm_expected = [0, 0, 1, 0]
        warning_actual = fs.calc_memberships(1.0)
        alarm_actual = fs.calc_memberships(2.0)
        self.assertEqual(warning_actual, warning_expected)
        self.assertEqual(alarm_actual, alarm_expected)

    def test_fullMembership_decreasing(self):
        fs = infer.FuzzyStates(3, 2, 1, 0)
        warning_expected = [0, 1, 0, 0]
        alarm_expected = [0, 0, 1, 0]
        warning_actual = fs.calc_memberships(2.0)
        alarm_actual = fs.calc_memberships(1.0)
        self.assertEqual(warning_actual, warning_expected)
        self.assertEqual(alarm_actual, alarm_expected)

    def test_50Membership_increasing(self):
        fs = infer.FuzzyStates(0, 1, 2, 3)
        warning_expected = [0, 0.5, 0.5, 0]
        alarm_expected = [0, 0, 0.5, 0.5]
        warning_actual = fs.calc_memberships(1.5)
        alarm_actual = fs.calc_memberships(2.5)
        self.assertEqual(warning_actual, warning_expected)
        self.assertEqual(alarm_actual, alarm_expected)

    def test_50Membership_decreasing(self):
        fs = infer.FuzzyStates(3, 2, 1, 0)
        warning_expected = [0, 0.5, 0.5, 0]
        alarm_expected = [0, 0, 0.5, 0.5]
        warning_actual = fs.calc_memberships(1.5)
        alarm_actual = fs.calc_memberships(0.5)
        self.assertEqual(warning_actual, warning_expected)
        self.assertEqual(alarm_actual, alarm_expected)


class TestEvidence(unittest.TestCase):
    def test_evidenceConstruction(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        midpoints = [0, 1, 2, 3]
        ev = infer.Evidence(thresh, cpt)
        self.assertEqual(ev.cpt, cpt)
        self.assertEqual(ev.thresholds, thresh)
        self.assertEqual(ev.states.normal, midpoints[0])
        self.assertEqual(ev.states.warning, midpoints[1])
        self.assertEqual(ev.states.alarm, midpoints[2])
        self.assertEqual(ev.states.danger, midpoints[3])

    def test_evidence_calcBiGivenA_test1(self):
        cpt = [0.02, 0.15, 0.35, 0.48]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        normal_actual = ev.calc_bi_given_a(0)
        warning_actual = ev.calc_bi_given_a(1)
        alarm_actual = ev.calc_bi_given_a(2)
        danger_actual = ev.calc_bi_given_a(3)
        self.assertEqual(cpt[0], normal_actual)
        self.assertEqual(cpt[1], warning_actual)
        self.assertEqual(cpt[2], alarm_actual)
        self.assertEqual(cpt[3], danger_actual)

    def test_evidence_calcBiGivenA_test2(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        normal_actual = ev.calc_bi_given_a(0)
        warning_actual = ev.calc_bi_given_a(1)
        alarm_actual = ev.calc_bi_given_a(2)
        danger_actual = ev.calc_bi_given_a(3)
        self.assertEqual(cpt[0], normal_actual)
        self.assertEqual(cpt[1], warning_actual)
        self.assertEqual(cpt[2], alarm_actual)
        self.assertEqual(cpt[3], danger_actual)

    def test_evidence_calcBiGivenNotA_test1(self):
        cpt = [0.02, 0.15, 0.35, 0.48]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        normal_actual = ev.calc_bi_given_not_a(0)
        warning_actual = ev.calc_bi_given_not_a(1)
        alarm_actual = ev.calc_bi_given_not_a(2)
        danger_actual = ev.calc_bi_given_not_a(3)
        self.assertEqual(cpt[3], normal_actual)
        self.assertEqual(cpt[2], warning_actual)
        self.assertEqual(cpt[1], alarm_actual)
        self.assertEqual(cpt[0], danger_actual)

    def test_evidence_calcBiGivenNotA_test2(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        normal_actual = ev.calc_bi_given_not_a(0)
        warning_actual = ev.calc_bi_given_not_a(1)
        alarm_actual = ev.calc_bi_given_not_a(2)
        danger_actual = ev.calc_bi_given_not_a(3)
        self.assertEqual(cpt[3], normal_actual)
        self.assertEqual(cpt[2], warning_actual)
        self.assertEqual(cpt[1], alarm_actual)
        self.assertEqual(cpt[0], danger_actual)


class TestContraryEvidence(unittest.TestCase):
    def test_contraryEvidenceConstruction(self):
        priors = [0.1, 0.2, 0.3, 0.4]
        thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(thresh, priors)
        self.assertEqual(cev.priors, priors)
        self.assertEqual(cev.states, thresh)

    def test_contraryEvidence_direction(self):
        priors = [0.1, 0.2, 0.3, 0.4]
        thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(thresh, priors)
        self.assertTrue(cev.direction_increasing)
        thresh2 = [3, 2, 1, 0]
        cev = infer.ContraryEvidence(thresh2, priors)
        self.assertTrue(~cev.direction_increasing)

    def test_contraryEvidence_calculate_prior_increasing(self):
        priors = [0.1, 0.2, 0.3, 0.4]
        thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(thresh, priors)
        normal_value = cev.calculate_prior(0.2)
        warning_value = cev.calculate_prior(1.2)
        alarm_value = cev.calculate_prior(2.2)
        danger_value = cev.calculate_prior(3.2)
        self.assertEqual(priors[0], normal_value)
        self.assertEqual(priors[1], warning_value)
        self.assertEqual(priors[2], alarm_value)
        self.assertEqual(priors[3], danger_value)

    def test_contraryEvidence_calculate_prior_decreasing(self):
        priors = [0.1, 0.2, 0.3, 0.4]
        thresh = [3, 2, 1, 0]
        cev = infer.ContraryEvidence(thresh, priors)
        normal_value = cev.calculate_prior(2.2)
        warning_value = cev.calculate_prior(1.2)
        alarm_value = cev.calculate_prior(0.2)
        danger_value = cev.calculate_prior(-0.2)
        self.assertEqual(priors[0], normal_value)
        self.assertEqual(priors[1], warning_value)
        self.assertEqual(priors[2], alarm_value)
        self.assertEqual(priors[3], danger_value)


class TestFailureMode(unittest.TestCase):
    def test_failureMode_construction(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev])
        self.assertEqual(fm.prior, 0.2)

    def test_calcProbBGivenA_1Node(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev])
        normal_actual = fm._prob_b_given_a([0])
        warning_actual = fm._prob_b_given_a([1])
        alarm_actual = fm._prob_b_given_a([2])
        danger_actual = fm._prob_b_given_a([3])
        self.assertEqual(cpt[0], normal_actual)
        self.assertEqual(cpt[1], warning_actual)
        self.assertEqual(cpt[2], alarm_actual)
        self.assertEqual(cpt[3], danger_actual)

    def test_calcProbBGivenA_2Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2])
        normal_actual = fm._prob_b_given_a([0, 0])
        warning_actual = fm._prob_b_given_a([1, 1])
        alarm_actual = fm._prob_b_given_a([2, 2])
        danger_actual = fm._prob_b_given_a([3, 3])
        self.assertEqual(0.01, normal_actual)
        self.assertEqual(0.04, warning_actual)
        self.assertEqual(0.09, alarm_actual)
        self.assertEqual(0.16, danger_actual)

    def test_calcProbBGivenA_3Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3])
        normal_actual = fm._prob_b_given_a([0, 0, 0])
        warning_actual = fm._prob_b_given_a([1, 1, 1])
        alarm_actual = fm._prob_b_given_a([2, 2, 2])
        danger_actual = fm._prob_b_given_a([3, 3, 3])
        self.assertEqual(0.001, normal_actual)
        self.assertEqual(0.008, warning_actual)
        self.assertEqual(0.027, alarm_actual)
        self.assertEqual(0.064, danger_actual)

    def test_calcProbBGivenNotA_1Node(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev])
        normal_actual = fm._prob_b_given_not_a([0])
        warning_actual = fm._prob_b_given_not_a([1])
        alarm_actual = fm._prob_b_given_not_a([2])
        danger_actual = fm._prob_b_given_not_a([3])
        self.assertEqual(cpt[3], normal_actual)
        self.assertEqual(cpt[2], warning_actual)
        self.assertEqual(cpt[1], alarm_actual)
        self.assertEqual(cpt[0], danger_actual)

    def test_calcProbBGivenNotA_2Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2])
        normal_actual = fm._prob_b_given_not_a([0, 0])
        warning_actual = fm._prob_b_given_not_a([1, 1])
        alarm_actual = fm._prob_b_given_not_a([2, 2])
        danger_actual = fm._prob_b_given_not_a([3, 3])
        self.assertEqual(0.16, normal_actual)
        self.assertEqual(0.09, warning_actual)
        self.assertEqual(0.04, alarm_actual)
        self.assertEqual(0.01, danger_actual)

    def test_calcProbBGivenNotA_3Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3])
        normal_actual = fm._prob_b_given_not_a([0, 0, 0])
        warning_actual = fm._prob_b_given_not_a([1, 1, 1])
        alarm_actual = fm._prob_b_given_not_a([2, 2, 2])
        danger_actual = fm._prob_b_given_not_a([3, 3, 3])
        self.assertEqual(0.064, normal_actual)
        self.assertEqual(0.027, warning_actual)
        self.assertEqual(0.008, alarm_actual)
        self.assertEqual(0.001, danger_actual)

    def test_calcProbB_1Node(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev])
        normal_actual = fm._prob_b([0])
        warning_actual = fm._prob_b([1])
        alarm_actual = fm._prob_b([2])
        danger_actual = fm._prob_b([3])
        self.assertEqual(0.34, normal_actual)
        self.assertEqual(0.28, warning_actual)
        self.assertEqual(0.22, alarm_actual)
        self.assertEqual(0.16, danger_actual)

    def test_calcProbB_2Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2])
        normal_actual = fm._prob_b([0, 0])
        warning_actual = fm._prob_b([1, 1])
        alarm_actual = fm._prob_b([2, 2])
        danger_actual = fm._prob_b([3, 3])
        self.assertEqual(0.13, normal_actual)
        self.assertEqual(0.08, warning_actual)
        self.assertEqual(0.05, alarm_actual)
        self.assertEqual(0.04, danger_actual)

    def test_calcProbB_3Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3])
        normal_actual = fm._prob_b([0, 0, 0])
        warning_actual = fm._prob_b([1, 1, 1])
        alarm_actual = fm._prob_b([2, 2, 2])
        danger_actual = fm._prob_b([3, 3, 3])
        self.assertEqual(0.0514, normal_actual)
        self.assertEqual(0.0232, warning_actual)
        self.assertEqual(0.0118, alarm_actual)
        self.assertEqual(0.0136, danger_actual)

    def test_calcProbAGivenB_1Node(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev])
        normal_actual = fm.calc_prob_a_given_b([0])
        warning_actual = fm.calc_prob_a_given_b([1])
        alarm_actual = fm.calc_prob_a_given_b([2])
        danger_actual = fm.calc_prob_a_given_b([3])
        self.assertEqual(0.05882, normal_actual)
        self.assertEqual(0.14286, warning_actual)
        self.assertEqual(0.27273, alarm_actual)
        self.assertEqual(0.50000, danger_actual)

    def test_calcProbAGivenB_2Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2])
        normal_actual = fm.calc_prob_a_given_b([0, 0])
        warning_actual = fm.calc_prob_a_given_b([1, 1])
        alarm_actual = fm.calc_prob_a_given_b([2, 2])
        danger_actual = fm.calc_prob_a_given_b([3, 3])
        self.assertEqual(0.01538, normal_actual)
        self.assertEqual(0.1, warning_actual)
        self.assertEqual(0.36, alarm_actual)
        self.assertEqual(0.8, danger_actual)

    def test_calcProbAGivenB_3Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3])
        normal_actual = fm.calc_prob_a_given_b([0, 0, 0])
        warning_actual = fm.calc_prob_a_given_b([1, 1, 1])
        alarm_actual = fm.calc_prob_a_given_b([2, 2, 2])
        danger_actual = fm.calc_prob_a_given_b([3, 3, 3])
        self.assertEqual(0.00389, normal_actual)
        self.assertEqual(0.06897, warning_actual)
        self.assertEqual(0.45763, alarm_actual)
        self.assertEqual(0.94118, danger_actual)

    def test_calcProbAGivenB_badLength(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3])
        with self.assertRaises(AssertionError):
            normal_actual = fm.calc_prob_a_given_b([0, 0])

    def test_calcProbAGivenB_Contrary_1Node(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev = infer.Evidence(thresh, cpt)
        cev_prior = [0.3, 0.2, 0.1, 0.05]
        cev_thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(cev_thresh, cev_prior)
        fm = infer.FailureMode(0.2, evidence=[ev], contrary=cev)
        normal_actual = fm.calc_prob_a_given_b([0], contrary=0.5)
        warning_actual = fm.calc_prob_a_given_b([1], contrary=1.5)
        alarm_actual = fm.calc_prob_a_given_b([2], contrary=2.5)
        danger_actual = fm.calc_prob_a_given_b([3], contrary=3.5)
        self.assertEqual(0.09677, normal_actual)
        self.assertEqual(0.14286, warning_actual)
        self.assertEqual(0.14286, alarm_actual)
        self.assertEqual(0.17391, danger_actual)

    def test_calcProbAGivenB_Contrary_2Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        cev_prior = [0.3, 0.2, 0.1, 0.05]
        cev_thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(cev_thresh, cev_prior)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2], contrary=cev)
        normal_actual = fm.calc_prob_a_given_b([0, 0], contrary=0.5)
        warning_actual = fm.calc_prob_a_given_b([1, 1], contrary=1.5)
        alarm_actual = fm.calc_prob_a_given_b([2, 2], contrary=2.5)
        danger_actual = fm.calc_prob_a_given_b([3, 3], contrary=3.5)
        self.assertEqual(0.02609, normal_actual)
        self.assertEqual(0.1, warning_actual)
        self.assertEqual(0.2, alarm_actual)
        self.assertEqual(0.45714, danger_actual)

    def test_calcProbAGivenB_Contrary_3Nodes(self):
        cpt = [0.1, 0.2, 0.3, 0.4]
        thresh = [-0.5, 0.5, 1.5, 2.5, 3.5]
        ev1 = infer.Evidence(thresh, cpt)
        ev2 = infer.Evidence(thresh, cpt)
        ev3 = infer.Evidence(thresh, cpt)
        cev_prior = [0.3, 0.2, 0.1, 0.05]
        cev_thresh = [0, 1, 2, 3]
        cev = infer.ContraryEvidence(cev_thresh, cev_prior)
        fm = infer.FailureMode(0.2, evidence=[ev1, ev2, ev3], contrary=cev)
        normal_actual = fm.calc_prob_a_given_b([0, 0, 0], contrary=0.5)
        warning_actual = fm.calc_prob_a_given_b([1, 1, 1], contrary=1.5)
        alarm_actual = fm.calc_prob_a_given_b([2, 2, 2], contrary=2.5)
        danger_actual = fm.calc_prob_a_given_b([3, 3, 3], contrary=3.5)
        self.assertEqual(0.00665, normal_actual)
        self.assertEqual(0.06897, warning_actual)
        self.assertEqual(0.27273, alarm_actual)
        self.assertEqual(0.77108, danger_actual)

if __name__ == '__main__':
    unittest.main()

