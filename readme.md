<h1 align="center">⚕️ Interpretable Clinical Decision Rules for Pediatric Cervical Spine Injuries⚕️️</h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.8-blue">
  <a href="https://github.com/Yu-Group/rule-vetting/actions"><img src="https://github.com/Yu-Group/rule-vetting/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-Group/rule-vetting/master">
</p>  

This is a *collaborative* repository intended to validate and derive clinical-decision rules for recognizing pediatric cervical spine injuries (CSIs). This project is work by Yaxuan Huang, Ishaan Srivastava, and William Torous, all in the UC Berkeley statistics department. Domain expertise is provided by Dr. Gabriel Devlin of UCSF.

We use a robust pipeline to vet previous clinical decision rules proposed in the academic literature and by hosptials. Additionally, we hope to externally validate the rules under study here with data from UCSF.

# Dataset

| Dataset |  Task                                                        | Size                            | References | Processed |
| ---------- | ----- | ----------------------------------------------------------- | :-------------------------------: | :--: |
|[csi_pecarn](rulevetting/projects/csi_pecarn)| Predict cervical spine injury in children | 3,314 patients, 540 with CSI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) |❌|


<p align="center">
    Research paper 📄, Data download link 🔗 
</br>
</p>

# Contributing checklist

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
	- [x] Rewrite the functions in `dataset.py` for processing the new dataset (e.g. see the dataset for [iai_pecarn](rulevetting/projects/iai_pecarn/dataset.py))
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
