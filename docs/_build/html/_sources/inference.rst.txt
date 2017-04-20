====================
 4. Inference
====================
Inference can be accomplished using time-series evidence within this 
PyDePr module.

* Assign evidence to a conclusion
* Use fuzzy logic to interpolate between conclusions

Evidence
---------
Conclusions are made using Bayesian inference based on the assigned Evidence.
This class defines an Evidence input to a Conclusion. Any symptom which indicates 
the Conclusion may be used as an Evidence input.

The following parameters may be supplied:

* thresholds: A list of thresholds in order: [normal, warning, alarm, danger, extreme danger]
* cpt: the conditional probability table for the evidence, which specifies the marginal probability distribution of the truth of the failure mode, depending on selected state. Must be an array of length 4. For example: [0.1, 0.2, 0.3, 0.4]
* name: an optional description or identifier.
* default_state: the default state of the evidence if bad input is received. 0 defaults to the prior, 1 defaults to Normal, 2 defaults to Warning, 3 defaults to Alarm, and 4 defaults to Danger. Recommended values are either 0 or 1.

Counter Evidence
-----------------
This class defines a contrary evidence input to a Conclusion. Any symptom which will 
excuplate the occurrence of a Conclusion can be used as a ContraryEvidence node.

The following parameters may be supplied:

* thresholds: A list of 4 thresholds in order: [normal, undetermined, abnormal, contradictory]
* priors: an array of updated priors for the Conclusion, based on the selected ContraryEvidence state. Must be length 4.
* name: an optional description or identifier.

Conclusion
-----------
This is the primary class that will be used in this model. Defines a Conclusion 
in terms of evidence and counter evidence.

The following parameters may be supplied:

* name: an optional description or identifier
* prior: the prior probability of occurrence
* evidence: an array of "Evidence" class nodes which define symptoms associated with the Conclusion. At least 1 must exist.
* contrary: a single "ContraryEvidence" class node which contradicts the Conclusion. A maximum of 1 may exist.

FuzzyStates
------------
This class is not meant to be used standalone, but it provides backend functionality
for the above classes.

FuzzyStates uses a fuzzy linear interpolation between the four user-defined states. 
Warning- the supplied values must be the "midpoints", or the thresholds at which
the fuzzy membership for the state should be equal to 100%. Total membership across 
all four states must always be equal to 100%.

The following parameters may be supplied:

* normal: signal value under normal behavior.
* warning: signal value that begins to indicate an anomaly.
* alarm: onset of significant poor behavior.
* danger: signal strongly indicates the Conclusion.
* steps: an optional parameter for the # of steps of the range of fuzzy values. Must be at least 100.