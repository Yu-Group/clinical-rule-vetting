# tbi_pecarn

**Data download 🔗**: https://pecarn.org/datasets/ "Identification of children at very low risk of clinically-important brain injuries after head trauma: a prospective cohort study."

**Paper link 📄**: https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf

**Prediction Task and Motivation**

The primary task of this project is to build on the prediction algorithm developed by Kupperman and his colleagues to predict whether a child who has been
admitted to the hospital for head trauma will be at low risk of having clinically important traumatic brain injury (TBI) given a certain set of covariates. 
As a primary cause of death among children, TBIs are crucial to diagnose. TBIs are commonly diagnosed via a CT scan, and while abnormalities on CT scans
can indicate TBIs, there is a growing concern that CT scans are being performed when not needed. In fact, according to Kupperman et al., the majority of CT scans 
performed are in individuals with minor head trauma, defined as a GCS score of 14 or 15, yet less than 10% of these CT scans actually show a TBI. Performing
CT scans on any individual poses the risk of radiation-induced malignancies, so we should be sparing in its use, especially for individuals that are likely to not have a TBI, where the risk of performing the CT scan might outweigh knowledge of the results. If we are able to predict whether a child is at low risk of having
a TBI, then we are able to identify a child for which a CT scan would be unnecessary.

**Dataset**

This dataset consists of children younger than 18 years of age who presented to one of the 25 hospitals in the PECARN hospital network, within 24 hours of some
head trauma. Since we are interested in predicting low risk of TBI for individuals where CT scans could possibly be avoided, we limit our analysis to individuals that have Glasgow Coma Scale (GCS) scores of 14 or 15, and were not pharmacologically sedated, pharmacologically paralyzed, or intubated at the time
of evaluation (the presence of the former variables might indicate a severe injury for which a CT scan would always be appropriate). 

For each child, information is provided for many variables, such as whether they have amnesia for the event, whether there is a palpable skull fracture, or
whether they have an altered mental state. Each individual's data was collected by a physician or trained site investigator who performed examinations
and answered corresponding questions on standardized data collection sheets. The sheets were then scanned to produce the provided database. Whether or not the
child had a clinically important TBI was then tabulated for each patient. The definition of ciTBI used in this analysis is a slight variation of the outcome 
used in Kupperman et al.: an individual has a ciTBI if they die in the emergency department, are intubated for more than 24 hours, are hospitalized for 2 or
more nights, underwent neurosurgery, or were marked as having a ciTBI by Kupperman et al. To ensure there were no missed ciTBI diagnoses after children were
discharged from the hospital, regular follow-ups were made to the parents and the correct outcome was imputed accordingly.

**Future Settings**

If a child presents to the emergency room with head trauma, and it is
unclear if a CT scan will be necessary to identify ciTBI (has GCS score as 14 or 15, not pharmacologically paralyzed, etc.), 
the physician can examine the patient and collect data on various covariates of importance to our classifier. We can then 
use these covariate values in our classifier to determine if the child is at low risk of ciTBI, and thus whether we can avoid
giving them a CT scan.

**Authors/Affiliations**

Jimmy Butler (UC Berkeley Department of Statistics)

Andrej Leban (UC Berkeley Department of Statistics)

Ian Shen (UC Berkeley Department of Statistics)

Xin Zhou (UC Berkeley Department of Biostatistics)

**Repository Directory**

+ baseline.py: Python script to implement and run baseline classifier
+ contributions.md: a list of contributions made by each group member
+ data_dictionary.md: a dictionary containing information about the final dataset and its variables
+ dataset.py: Python script to perform cleaning, preprocessing, and feature extraction
+ figs: folder of figures used in the writeup
+ helper.py: Python script containing helper functions for dataset.py
+ model_best.pkl: relevant stats for our best model (saved)
+ model_best.py: Python script to run the best model
+ notebooks: folder containing jupyter notebooks with analyses used in our writeup (see folder for readme of which notebooks were used in which part of the writeup analysis)
+ readme.md: this file
+ references.bib: references file for the writeup
+ tbi_df.rmd: documented analysis of random forests used in report
+ tbi_df.pdf: documented analysis of random forests used in report
+ writeup.rmd: writeup
+ writeup.pdf: writeup
