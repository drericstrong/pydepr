====================
 1. Introduction
====================
PyDePr is a toolkit designed to facilitate the preprocessing and validation of 
machine learning models. What does that mean? Machine learning models are often 
specialized, requiring years of experience to correctly configure. The hardest 
step is often preprocessing the dataset to be analyzed. An old joke in machine
learning is that 90% of the work is cleaning the dataset, while only 10% is 
actual data analysis.

Keeping that in mind, PyDePr is meant to ease the initial step of model-building
by automatically preprocessing some very specific types of machine learning models.
PyDePr is not meant to build machine learning models for you (there are plenty of
Python libraries for that purpose), and it is not meant to be some sort of  
catch-all solution for all types of machine learning models. However, support for
additional model types will be added as PyDePr is developed. 

Package Organization
----------------------
PyDePr is organized into several namespaces, including:

* regression
* inference

The regression namespace handles all preprocessing and validation for regression 
models, including a visualization of results. The inference namespace is meant
for Bayesian inference, where accumulated evidence points to a conclusion.