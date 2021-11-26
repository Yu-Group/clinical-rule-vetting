<h1 align="center">⚕️ Interpretable Clinical Decision Rules ⚕️️</h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7-blue">
  <a href="https://github.com/Yu-Group/rule-vetting/actions"><img src="https://github.com/Yu-Group/rule-vetting/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-Group/rule-vetting/master">
</p>  

This is a *collaborative* repository intended to validate and derive clinical-decision rules. We use a unified  pipeline across a variety of contributed datasets to vet previous modeling practices for clinical decision rules. Additionally, we hope to externally validate the rules under study here with data from UCSF.

# Rule derivation datasets

| Dataset |  Task                                                        | Size                            | References | Processed |
| ---------- | ----- | ----------------------------------------------------------- | :-------------------------------: | :--: |
|[iai_pecarn](rulevetting/projects/iai_pecarn)| Predict intra-abdominal injury requiring acute intervention before CT | 12,044 patients, 203 with IAI-I | [📄](https://pubmed.ncbi.nlm.nih.gov/23375510/), [🔗](https://pecarn.org/datasets/) | ✅ |
|[tbi_pecarn](rulevetting/projects/tbi_pecarn)| Predict traumatic brain injuries before CT | 42,412 patients, 376 with ciTBI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) | ❌ |
|[csi_pecarn](rulevetting/projects/csi_pecarn)| Predict cervical spine injury in children | 3,314 patients, 540 with CSI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) |❌|
|[tig_pecarn](rulevetting/projects/tig_pecarn)| Predict bacterial/non-bacterial infections in febrile infants from RNA transcriptional biosignatures | 279 patients, ? with infection | [🔗](https://pecarn.org/datasets/) |❌|
|exxagerate| Predict 30-day mortality for acute exacerbations of chronic obstructive pulmonary disease (AECOPD) | 1,696 patients, 17 mortalities | [📄](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8354163/), [🔗](https://data.mendeley.com/datasets/3868pbf375/2) |❌|
|heart_disease_uci| Predict heart disease presence from basic attributes / screening | 920 patients, 509 with heart disease | [📄](https://www.sciencedirect.com/science/article/abs/pii/0002914989905249), [🔗](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |❌|



<p align="center">
    Research paper 📄, Data download link 🔗 
</br>
</p>

Datasets are all tabular (or at least have interpretable input features), reasonably large (e.g. have at least 100 positive and negative cases), and have a binary outcome. For PECARN datasets, please read and agree to the research data use agreement on the [PECARN website](https://pecarn.org/datasets/).

**Possible data sources**: [PECARN datasets](https://pecarn.org/datasets/) |  [Kaggle datasets](https://www.kaggle.com/search?q=healthcare+tag%3A%22healthcare%22) | [MDCalc](https://www.mdcalc.com/) | [UCI](https://archive.ics.uci.edu/ml/index.php) | [OpenML](https://www.openml.org/home) | [MIMIC](https://physionet.org/content/mimiciv/1.0/) | [UCSF De-ID](https://data.ucsf.edu/research/deid-data)
**Potential specific datasets**: Maybe later will expand to other high-stakes datasets (e.g. COMPAS, loan risk).

# Contributing checklist

To contribute a new project (e.g. a new dataset + modeling), create a pull request following the steps below. The easiest way to do this is to copy-paste an existing project (e.g. [iai_pecarn](rulevetting/projects/iai_pecarn)) into a new folder and then edit that one.

Helpful docs: [Collaboration details](docs/collaborating_for_data_scientists.md) | [Lab writeup](docs/lab_writeup.md) | [Slides](https://rules.csinva.io/pres/index.html#/)

- [x] Repo set up
  - [x] Create a fork of this repo (see tutorial on forking/merging [here](https://jarv.is/notes/how-to-pull-request-fork-github/))
  - [x] Install the repo as shown [below](https://github.com/Yu-Group/rule-vetting#installation)
  - [x] Select a dataset - once you've selected, open an issue in this repo with the name of the dataset + a brief description so others don't work on the same dataset 	
  - [x] Assign a `project_name` to the new project (e.g. `iai_pecarn`) 	
- [ ] Data preprocessing
  - [x] Download the raw data into `data/{project_name}/raw`
    - Don't commit any very large files
  - [x] Copy the template files from `rulevetting/projects/iai_pecarn` to a new folder `rulevetting/projects/{project_name}`
	- [ ] Rewrite the functions in `dataset.py` for processing the new dataset (e.g. see the dataset for [iai_pecarn](rulevetting/projects/iai_pecarn/dataset.py))
    - [ ] Document any judgement calls you aren't sure about using the `dataset.get_judgement_calls_dictionary` function
        - See [the template file](rulevetting/templates/dataset.py) for documentation of each function or the [API documentation](https://yu-group.github.io/rule-vetting/)
    - Notebooks / helper functions are optional, all files should be within `rulevetting/projects/{project_name}`
- [ ] Data description
  - [ ] Describe each feature in the processed data in a file named `data_dictionary.md`
  - [ ] Summarize the data and the prediction task in a file named `readme.md`. This should include basic details of data collection (who, how, when, where), why the task is important, and how a clinical decision rule may be used in this context. Should also include your names/affiliations.
- [ ] Modeling
  - [ ] Baseline model - implement `baseline.py` for predicting given a baseline rule (e.g. from the existing paper)
    - should override the [model template](rulevetting/templates/model.py) in a class named `Baseline`
  - [ ] New model - implement `model_best.py` for making predictions using your newly derived best model
    - also should override the [model template](rulevetting/templates/model.py) in a class named `Model`
- [ ] Lab writeup (see [instructions](docs/lab_writeup.md))    
  - [ ] Save writeup into `writeup.pdf` + include source files
  - Should contain details on exploratory analysis, modeling, validation, comparisons with baseline, etc.
- [ ] Submitting
  - [ ] Ensure that all tests pass by running `pytest --project {project_name}` from the repo directory
  - [ ] [Open a pull request](https://jarv.is/notes/how-to-pull-request-fork-github/) and it will be reviewed / merged
- [ ] Reviewing submissions
  - [ ] Each pull request will be reviewed by others before being merged



# Installation

Note: requires python 3.7 and pytest (for running the automated tests). 
It is best practice to create a [venv](https://docs.python.org/3/tutorial/venv.html) or pipenv for this project.

```bash
python -m venv rule-env
source rule-env/bin/activate
```

Then, clone the repo and install the package and its dependencies.

```bash
git clone https://github.com/Yu-Group/rule-vetting
cd rule-vetting
pip install -e .
```

Now run the automatic tests to ensure everything works.

```bash
pytest --project iai_pecarn
```

To use with jupyter, might have to add this venv as a jupyter kernel.

```bash
python -m ipykernel install --user --name=rule-env
```

# Clinical Trial Datasets

| Dataset |  Task                                                        | Size                            | References | Processed |
| ---------- | ----- | ----------------------------------------------------------- | :-------------------------------: | :--: |
|bronch_pecarn| Effectiveness of oral dexamethasone for acute bronchiolitisintra-abdominal injury requiring acute intervention before CT | 600 patients, 50% control | [📄](https://pubmed.ncbi.nlm.nih.gov/17652648/), [🔗](https://pecarn.org/datasets/) | ❌ |
| gastro_pecarn | Impact of Emergency Department Probiotic Treatment of Pediatric Gastroenteritis | 886 patients, 50% control | [📄](https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-15-170), [🔗](https://pecarn.org/datasets/) | ❌ |



<p align="center">
    Research paper 📄, Data download link 🔗 
</br>
</p>

# Reference
<details>
<summary>Background reading</summary>
<ul>
  <li>Be familiar with the <a href="https://github.com/csinva/imodels">imodels</a>: package</li>
  <li>See the <a href="https://www.equator-network.org/reporting-guidelines/tripod-statement/">TRIPOD</a> statement on medical reporting</li>
  <li>See the <a href="https://www.pnas.org/content/117/8/3920">Veridical data science</a> paper</li>
</ul>
</details>

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
