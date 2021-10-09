<h1 align="center">Interpretable Clinical Decision Rules ⚕️ </h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.9-blue">
 </p>  
This is a *collaborative* repository intended to validate and derive clinical-decision rules. 





# Installation

Note: requires python 3.6-3.9 and pytest (for running the automated tests). 
It is best practice to create a virtualenv or pipenv for this project.

```bash
git clone https://github.com/Yu-Group/medical-rules
cd medical-rules
python setup.py sdist
pip install -e .
```

Now run the automatic tests to ensure everything works.

```
pytest
```



# How do I contribute?

To contribute, fork this repo + create a pull request for a new project.


- data preprocessing
	- extract out reliable features / feature names
	- impute missing values correctly
	- discretization
	- feature selection - remove things which are toooo correlated
	- create data dictionary
- modeling
	- build and test the original model
	- run many models from imodels
	- extract out stable rules: screen for high predictive acc, look at what is kept
	- build stable rules model (e.g. using RuleFit or Corels)
- validation
	- predictive accuracy
	- number of rules
	- overlap with original rules?
- test
	- run on UCSF held-out data?
- documentation is available in the [API](yu-group.github.io/medical-rules/)


# Datasets

| Dataset id | Task                                                        | Size                            |
| ---------- | ----------------------------------------------------------- | ------------------------------- |
| [iai_pecarn](projects/iai_pecarn) | Predict intra-abdominal injury requiring acute intervention | 12,044 patients, 203 with IAI-I |
|            |                                                             |                                 |
|            |                                                             |                                 |



- currently in the process of downloading / preprocessing a series of tabular datasets
  - see [PECARN datasets](https://pecarn.org/datasets/)
  - see [Kaggle datasets](https://www.kaggle.com/search?q=healthcare+tag%3A%22healthcare%22)
  - see rules in [MDCalc](https://www.mdcalc.com/)
  - open to many more datasets
- criteria for each dataset
  - must be reasonably large (e.g. have at least 100 positive and negative cases)
  - currently must be binary outcome
  - maybe should already have an existing rule

# Reference

- Inspired by the [BIG-bench](https://github.com/google/BIG-bench) effort