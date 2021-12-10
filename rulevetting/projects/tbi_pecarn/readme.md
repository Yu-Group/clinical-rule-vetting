# tbi_pecarn

Authors: Florica Constantine, Hyunsuk Kim, Mark Oussoren, Sahil Saxena

Affiliation: University of Califoria, Berkeley Statistics

<!-- Summarize the data and the prediction task: basic details of data collection (who, how, when, where), why the task is important, and how a clinical decision rule may be used in this context) -->

<!-- Sahil: -->
Task: Doctors from PECARN collected data on around 43,000 patients, all under age 18, who reported head trauma. The study was conducted at 25 pediatric emergency department across the United States from June, 2004 to September, 2006. Given that CT scans and other treatments can cause long-term adverse effects, especially in young children, this task aims to propose intensive treatments only to those patients for whom it is truly necessary. The updated and improved clinical decision rule (using a l1-regularized logistic regression model) may be used in emergency departments in this way: on-site physicians can record around 100 features of the patient (including patient history, injury mechanism, symptoms, and more) into our model (best_model.py), which will then output a determination of whether the patient has a clinically important traumatic brain injury or not. Note this determination should serve as a recommendation based on data from previous cases, but every case is unique. 

<!-- Current: -->
Task: To derive updated and improved decision rules relative to the paper below. 

**Data download 🔗**: https://pecarn.org/datasets/ "IDENTIFICATION OF CHILDREN AT VERY LOW RISK OF CLINICALLY-IMPORTANT BRAIN INJURIES AFTER HEAD TRAUMA: A PROSPECTIVE COHORT STUDY."

**Paper link 📄**: https://pecarn.org/studyDatasets/documents/Kuppermann_2009_The-Lancet_000.pdf

> **Abstract**
> 
> Study objective: The overall objective of this study is to develop a clinical decision rule for appropriate neuroimaging of children after minor-to-moderate head trauma. The goal of the study is to create a decision rule which identifies those children in need of emergent imaging (i.e. CT scan) and treatment, while reducing the use of head CT scans in those children with minimal risk of traumatic brain injuries.


Code for reproducing analysis evaluating the PECARN Clinical Decision rule for predicting TBI in children with minor head trauma. 

>  **Study Period:** June 2004 - September 2006
>
>  **Study Type:** Observational
>
> **Study Enrollment:** 42412
>
> **Consent:** Waiver of consent at some sites; verbal consent for telephone follow up at other sites


