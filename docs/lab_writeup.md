*Some instructions for the Stat 215A final project writeup. See the [contributing checklist](https://github.com/Yu-Group/rule-vetting#contributing-checklist) for all deliverables and the [collaboration guide](https://github.com/Yu-Group/rule-vetting/docs/collaborating_for_data_scientists.md).*

# Writeup

The lab writeup should be inside the project folder (e.g. `rulevetting/projects/iai_pecarn/writeup.pdf`) and answer these questions in a file named `writeup.pdf`.
The accomanying source file `writeup.Rmd` (or `writeup.md`) should also be inside the project folder.
Figures should all be in a subdirectory named `figs`. If you use any packages outside of the ones already installed, please include a `requirements.txt` file with the name and version number of these packages.

Do write comments in the code - we will be checking these!


**Handy links**: Bin’s data wisdom for data science [blog post](http://www.odbms.org/2015/04/data-wisdom-for-data-science/)

## 1. Domain problem to solve

How can we best vet and/or improve the clinical decision rule for your given problem? Most importantly, the clinical decision rule should be highly predictive and minimize the amount of missed diagnoses (i.e. have a very high sensitivity). It should also be easy-to-use, using variables that clinicians can readily have access to when making their decisions. Finally, the interpretability of the rule helps to check whether its predictions will make sense for new patients and makes it easier to apply in new settings.

## 2. Data Collection
What are the most relevant data to collect to answer the question in (1)?

Ideas from experimental design (a subfield of statistics) and active learning (a subfield of machine learning) are useful here. The above question is good to ask even if the data has already been collected because understanding the ideal data collection process might reveal shortcomings of the actual data collection process and shed light on analysis steps to follow.

The questions below are useful to ask: How were the data collected? At what locations? Over what time period? Who collected them? What instruments were used? Have the operators and instruments changed over the period? Try to imagine yourself at the data collection site physically.
## 3. Meaning
What does each variable mean in the data? What does it measure? Does it measure what it is supposed to measure? How could things go wrong? What statistical assumptions is one making by assuming things didn’t go wrong? (Knowing the data collection process helps here.)

Meaning of each variable -- ask students to imagine being there at the ER and giving a Glasgow coma score, for example, and also a couple of variables -- ask students what could cause different values written down.

How were the data cleaned? By whom? 

## 4. Relevance
Can the data collected answer the substantive question(s) in whole or in part? If not, what other data should one collect? The points made in (2) are pertinent here.

## 5. Translation
How should one translate the question in (1) into a statistical question regarding the data to best answer the original question? Are there multiple translations? For example, can we translate the question into a prediction problem or an inference problem regarding a statistical model? List the pros and cons of each translation relative to answering the substantive question before choosing a model.

Do we have multiple reasonable translations? 
## 6. Comparability
Are the data units comparable or normalized so that they can be treated as if they were exchangeable? Or are apples and oranges being combined? Are the data units independent? Are two columns of data duplicates of the same variable?
## 7. Visualization
Look at data (or subsets of them). Create plots of 1- and 2-dimensional data. Examine summaries of such data. What are the ranges? Do they make sense? Are there any missing values? Use color and dynamic plots. Is anything unexpected? It is worth noting that 30% of our cortex is devoted to vision, so visualization is highly effective to discover patterns and unusual things in data. Often, to bring out patterns in big data, visualization is most useful after some model building, for example, to obtain residuals to visualize.
## 8. Randomness
Statistical inference concepts such as p-values and confidence intervals rely on randomness. What does randomness mean in the data? Make the randomness in the statistical model as explicit as possible. What domain knowledge supports such a statistical or mathematical abstraction or the randomness in a statistical model?

What is the randomness in this PECARN data set? Is it a random sample from a population? Which one? Why can the data be viewed as a random sample?What assumptions are being made? Can one check these conditions using the info on the data collection process?
## 9. Stability
What off-the-shelf method will you use? Do different methods give the same qualitative conclusion? Perturb one’s data, for example, by adding noise or subsampling if data units are exchangeable (in general, make sure the subsamples respect the underlying structures, e.g. dependence, clustering, heterogeneity, so the subsamples are representative of the original data). Do the conclusions still hold? Only trust those that pass the stability test, which is an easy-to-implement, first defense against over-fitting or too many false positive discoveries.

