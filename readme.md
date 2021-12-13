<h1 align="center">⚕️ Interpretable Clinical Decision Rules for Pediatric Cervical Spine Injuries⚕️️</h1>
<p align="center"> Validating and deriving clinical-decision rules. Work-in-progress.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7-blue">
  <a href="https://github.com/Yu-Group/rule-vetting/actions"><img src="https://github.com/Yu-Group/rule-vetting/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-Group/rule-vetting/master">
</p>  

This is a *collaborative* repository intended to validate and derive clinical decision rules for recognizing pediatric cervical spine (c-spine) injuries. This project is work by Yaxuan Huang, Ishaan Srivastava, and William Torous, all in the UC Berkeley statistics department. Answers to domain questions and suggestions from a practitioner's perspective are provided by Dr. Gabriel Devlin of UCSF.

A cervical spine injury refers to an injury in the first seven vertebrae (C1-C7) of the neck. These injuries are extremely important to diagnosis and treat correctly because they can damage the nervous system and lead to paralysis. Common mechanisms of c-spine injuries in children include motor vehicle accidents, falls from heights, and sports collisions. CT scans are a standard diagnostic intervention for adults suspected of having a c-spine injury. Due to differences in physiology, c-spine injuries are less common in children than adults, and less than $1 \%$ of pediatric trauma cases in the ED are caused by them (Leonard et. al (2011)). Because of this lower likelihood of injury and children's increased risk from radiation, the decision to use a CT scan must be carefully weighed against alternatives by ED doctors. 

The goal of this project is to use the PCS framework to robustly validate current clinical decision rules for c-spine injuries as well as to propose new ones. Decision rules are highly-interpretable data-driven tests for diagnosing certain conditions or suggesting certain interventions. These tests are generally built from data about previous medical cases and help doctors make more informed decisions. This project focuses on decision rules to diagnose any type of c-spine injury (e.g. fracture or ligament damage) during an initial ED examination. Providing ED doctors with more specific c-spine decision rules which do not sacrifice specificity will hopefully decrease the number of unnecessary diagnostic CT scans. Compared to other clinical decision rules, proposed c-spine injury rules have not been as thoroughly vetted by researchers. Leonard et. al (2011) abstract covariates from over 3000 emergency department (ED) visits for potential c-spine injuries and propose a baseline decision rule.

This project gratefully uses that study's data and compares results with its decision rule. The data comes from 17 hospitals in the Pediatric Emergency Care Applied Research Network (PECARN) over a four year period from 2000 to 2004. These hospitals are large urban hospitals which serve as regional trauma centers and which focus on pediatric care. Patients in the ED at these hospitals can arrive on their own, by EMS, or as transfers from other hospitals. Pediatric doctors extracted over 600 covariates from each patient's medical records, including hand-written comments by the attending doctor, at a later date. These covariates thoroughly describe a patient's condition upon initial evaluation at the ED (as well as by EMS and an outside hospital, if applicable), their prior medical history, information from any X-rays, CTs, or MRIs performed, and the outcomes of their visit. Leonard et. al summarize the most salient information into 32 binary indicators which are also provided. Leonard et al. also match each of the 540 patients with a c-spine injury to three control group units which do not: randomly, by mechanism of injury, and by EMS arrival. Additionally, the rules under study here will potentially be externally validated with data from UCSF.

Inspired by Leonard et. al's positive results with binary variables, this project frames the prediction task for c-spine injuries as a binary decision problem. This choice allows prioritization of the models' interpretability and ease of use of in a hectic ED environment. We verify our predictions with withheld information about the case's interventions and outcomes while also using demographic data to consider equity.

# Dataset

| Dataset |  Task                                                        | Size                            | References | Processed |
| ---------- | ----- | ----------------------------------------------------------- | :-------------------------------: | :--: |
|[csi_pecarn](rulevetting/projects/csi_pecarn)| Predict cervical spine injury in children | 3,314 patients, 540 with CSI | [📄](https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf), [🔗](https://pecarn.org/datasets/) |✅|


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
- [x] Data preprocessing
  - [x] Download the raw data into `data/{project_name}/raw`
    - Don't commit any very large files
  - [x] Copy the template files from `rulevetting/projects/iai_pecarn` to a new folder `rulevetting/projects/{project_name}`
	- [x] Rewrite the functions in `dataset.py` for processing the new dataset (e.g. see the dataset for [iai_pecarn](rulevetting/projects/iai_pecarn/dataset.py))
    - [x] Document any judgement calls you aren't sure about using the `dataset.get_judgement_calls_dictionary` function
        - See [the template file](rulevetting/templates/dataset.py) for documentation of each function or the [API documentation](https://yu-group.github.io/rule-vetting/)
    - Notebooks / helper functions are optional, all files should be within `rulevetting/projects/{project_name}`
- [x] Data description
  - [x] Describe each feature in the processed data in a file named `data_dictionary.md`
  - [x] Summarize the data and the prediction task in a file named `readme.md`. This should include basic details of data collection (who, how, when, where), why the task is important, and how a clinical decision rule may be used in this context. Should also include your names/affiliations.
- [x] Modeling
  - [x] Baseline model - implement `baseline.py` for predicting given a baseline rule (e.g. from the existing paper)
    - should override the [model template](rulevetting/templates/model.py) in a class named `Baseline`
  - [x] New model - implement `model_best.py` for making predictions using your newly derived best model
    - also should override the [model template](rulevetting/templates/model.py) in a class named `Model`
- [x] Lab writeup (see [instructions](docs/lab_writeup.md))    
  - [x] Save writeup into `writeup.pdf` + include source files
  - Should contain details on exploratory analysis, modeling, validation, comparisons with baseline, etc.
- [x] Submitting
  - [x] Ensure that all tests pass by running `pytest --project {project_name}` from the repo directory
  - [x] [Open a pull request](https://jarv.is/notes/how-to-pull-request-fork-github/) and it will be reviewed / merged
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

Now run the automatic tests to ensure everything works (warnings are fine as long as all test pass).

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
