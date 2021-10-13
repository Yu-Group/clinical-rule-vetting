<h1 align="center">⚕️ Interpretable Clinical Decision Rules ⚕️️</h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7-blue">
  <a href="https://github.com/Yu-Group/medical-rules/actions"><img src="https://github.com/Yu-Group/medical-rules/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-Group/medical-rules/master">
 </p>  
This is a *collaborative* repository intended to validate and derive clinical-decision rules.
We hope to use a unified modeling pipeline across a variety of contributed datasets to standardize and improve previous modeling practices for clinical decision rules.
Additionally, we hope to externally validate the rules under study here with data from UCSF.

# Datasets

| Dataset |  Task                                                        | Size                            | References | Processed |
| ---------- | ----- | ----------------------------------------------------------- | :-------------------------------: | :--: |
| [iai_pecarn](mrules/projects/iai_pecarn) | Predict intra-abdominal injury requiring acute intervention before CT | 12,044 patients, 203 with IAI-I | [📄](https://pubmed.ncbi.nlm.nih.gov/23375510/), [🔗](https://pecarn.org/datasets/) | ✅ |
|[tbi_pecarn](mrules/projects/tbi_pecarn)| Predict traumatic brain injuries before CT | 42,412 patients, 376 with ciTBI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) | ❌ |
|[csi_pecarn](mrules/projects/csi_pecarn)| Predict cervical spine injury in children | 3,314 patients, 540 with CSI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) |❌|
|[tig_pecarn](mrules/projects/tig_pecarn)| Predict bacterial/non-bacterifal infections in febrile infants from RNA transcriptional biosignatures | 279 patients, ? with infection | [🔗](https://pecarn.org/datasets/) |❌|

<p align="center">
    Research paper 📄, Data download link 🔗 
</br>
</p>
Datasets must be tabular (or at least have interpretable input features), be reasonably large (e.g. have at least 100 positive and negative cases), and have a binary outcome. If this goes well, might also expand to other high-stakes datasets (e.g. COMPAS, loan risk).

To use PECARN datasets, please read an agree to the research data use agreement on the [PECARN website](https://pecarn.org/datasets/).

Possible data sources: [PECARN datasets](https://pecarn.org/datasets/) |  [Kaggle datasets](https://www.kaggle.com/search?q=healthcare+tag%3A%22healthcare%22) | [MDCalc](https://www.mdcalc.com/) | [UCI](https://archive.ics.uci.edu/ml/index.php) ([heart disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)) | [OpenML](https://www.openml.org/home)



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
  - [ ] Copy over the template files from `mrules/projects/iai_pecarn` to a new folder `mrules/projects/{project_name}`
  - [ ] Rewrite the functions in `mrules/projects/{project_name}/dataset.py` for processing the new dataset (e.g. see the dataset for [iai_pecarn](mrules/projects/iai_pecarn/dataset.py))
	- Note: Notebooks / helper functions are optional
    - See [the template file](mrules/templates/dataset.py) for documentation of each function
- [ ] Data description
  - [ ] Add a `mrules/projects/{project_name}/data_dictionary.md` file that describes each feature in the processed data
  - [ ] Add a `mrules/projects/{project_name}/readme.md` file that describes the data and the prediction task. This should include basic details of data collection (who, how, when, where) and why it is important, and how a clinical decision rule may be used in this context. Should also include your names/affiliations.
- [ ] Modeling
  - [ ] Implement the functions in `mrules/projects/{project_name}/baseline.py` for predicting given a baseline rule (if there is no existing rule for this project, then have each method simply return None)
    - See [the template file](mrules/templates/baseline.py) for documentation of each function
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

Note: requires python 3.7 and pytest (for running the automated tests). 
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