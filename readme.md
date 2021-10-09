<h1 align="center">Interpretable Clinical Decision Rules ⚕️ </h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.9-blue">
  <a href="https://github.com/Yu-Group/medical-rules/actions"><img src="https://github.com/Yu-Group/medical-rules/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-Group/medical-rules/master">
 </p>  

This is a *collaborative* repository intended to validate and derive clinical-decision rules.
We hope to use a unified modeling pipeline across a variety of contributed datasets to standardize and improve previous modeling practices.


# Still under development
- `baseline.py` template
- modeling pipeline
- visualization / posthoc pipeline

# Datasets

| Dataset id | Task                                                        | Size                            |
| ---------- | ----------------------------------------------------------- | ------------------------------- |
| [iai_pecarn](projects/iai_pecarn) | Predict intra-abdominal injury requiring acute intervention | 12,044 patients, 203 with IAI-I |
|            |                                                             |                                 |
|            |                                                             |                                 |

Datasets must be tabular (or at least have interpretable input features),
be reasonably large (e.g. have at least 100 positive and negative cases),
and have a binary outcome.

Possible data sources: [PECARN datasets](https://pecarn.org/datasets/) |  [Kaggle datasets](https://www.kaggle.com/search?q=healthcare+tag%3A%22healthcare%22) | [MDCalc](https://www.mdcalc.com/)



# How do I contribute?

To contribute a new project (e.g. a new dataset + modeling), create a pull request following the steps belwo.
The easiest way to do this is to copy-paste an existing project (e.g. [iai_pecarn](mrules/projects/iai_pecarn)) into a new folder and then edit that one.

- [ ] Repo set up
  - [ ] Create a fork of this repo
  - [ ] Install the repo as shown above
  - [ ] Select a dataset - once you've selected, open an issue in this repo with the name of the dataset + a brief description so others don't work on the same dataset 	
  - [ ] Come up with a `project_name` for the new project (e.g. iai_pecarn) 	
- [ ] Data preprocessing
  - [ ] Download the raw data into `data/{project_name}/raw`
  - [ ] Rewrite the functions in `dataset.py` for processing the new dataset (e.g. see the dataset for [iai_pecarn](mrules/projects/iai_pecarn/dataset.py))
	- Note: Notebooks / helper functions are optional
- [ ] Data description
  - [ ] Add a `data_dictionary.md` file that describes each feature in the processed data
  - [ ] Add a `readme.md` file that describes the data and the prediction task. This should include basic details of data collection (who, how, when, where) and why it is important, and how a clinical decision rule may be used in this context
- [ ] Modeling
  - [ ] Implement the functions in `baseline.py` for predicting given a baseline rule (if there is no existing rule for this project, then leave these not implemented)
- [ ] Merging
  - [ ] Ensure that all tests pass by running `pytest` from the repo directory
  - [ ] Open a pull request and it will be reviewed / merged	


**What is included in each step.**

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

# Reference
<details>
<summary>Related packages</summary>
<ul>
  <li><a href="https://github.com/csinva/imodels">imodels</a>: rule-based modeling</li>
  <li><a href="https://github.com/Yu-Group/veridical-flow">veridical-flow</a>: stability-based analysis</li>
  <li><a href="https://github.com/trevorstephens/gplearn/tree/ad57cb18caafdb02cca861aea712f1bf3ed5016e">gplearn</a>: symbolic regression/classification</li>
  <li><a href="https://github.com/dswah/pyGAM">pygam</a>: generative additive models</li>
  <li><a href="https://github.com/interpretml/interpret">interpretml</a>: boosting-based gam</li>
</ul>
</details>

<details>
<summary>Updates</summary>
<ul>
  <li>For updates, star the repo, <a href="https://github.com/csinva/csinva.github.io">see this related repo</a>, or follow <a href="https://twitter.com/csinva_">@csinva_</a></li>
  <li>Please make sure to give authors of original datasets appropriate credit!</li>
  <li>Contributing: pull requests <a href="https://github.com/csinva/imodels/blob/master/docs/contributing.md">very welcome</a>!</li>
</ul>
</details>

<details>
<summary>Related open-source collaborations</summary>
<ul>
  <li>The <a href="https://github.com/csinva/imodels">imodels package</a> maintains many of the rule-based models here</li>
  <li>Inspired by the <a href="https://github.com/csinva/imodels">BIG-bench</a> effort.</li>
  <li>See also <a href="https://github.com/GEM-benchmark/NL-Augmenter">NL-Augmenter</a> and <a href="https://github.com/allenai/natural-instructions-expansion">NLI-Expansion</a></li>
</ul>
</details>