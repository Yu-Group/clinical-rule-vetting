<p align="center"> <i> Some instructions for clinicians aiding with rule-vetting. </i> </p>


Thank you for helping to provide us with your extremely helpful knowledge in vetting clinical decision rules! Your expertise is crucial for understanding, building, and vetting these CDRs.

In the course of this project, we would like each group to meet you twice for domain expertise in the problem, once in the week of 11/15 and once in the week of 11/29 - the group will reach out to you over email to schedule these meetings before Nov. 12 when they will get their assignment.

### Background

Outside of meetings, we hope your time-commitment would be low. If you are not familiar with your dataset/study, you might read it to get a refresher of the context it was collected in.

In case you are not familiar with the PCS framework, here are some links (the video is probably the most accessible before reading the paper)

- [Veridical data science](https://www.google.com/url?q=https%3A%2F%2Fwww.pnas.org%2Fcontent%2F117%2F8%2F3920&sa=D&sntz=1&usg=AFQjCNHxGUnnaw4Av_7kFoGiSLwqExbdEA) (PCS framework), PNAS, 2020 [(QnAs with Bin Yu)](https://www.google.com/url?q=https%3A%2F%2Fwww.pnas.org%2Fcontent%2F117%2F8%2F3893&sa=D&sntz=1&usg=AFQjCNEAQHjmn1HUGUI0l09tz5gjNPP9zw)
- [Breiman Lecture (video) at NeurIPS "Veridical data Science"](https://www.google.com/url?q=https%3A%2F%2Fslideslive.com%2F38922599%2Fveridical-data-science&sa=D&sntz=1&usg=AFQjCNHYmnQjvxfiLqMBmcmQJsCYE2hkeQ) 2019

###  Meeting 1

When cleaning the data, the group will need to translate from messy data to meaningful features. Your valuable expertise will help decide which variables to use + how to process them. It will also be helpful if you have as much information as possible on how the dataset was collected.

When meeting with your group, they will come prepared with a list of variables and questions about them.

> **Examples**
>
> - Who collected the data? Over what period? How?
> - Are these variables available at the time when the clinical decision rule is used?
> - Are the distributions of each variable reasonable?
> - Is this variable always available?
> - What does it mean for these variables to be "missing" - can they be imputed using other information? How and why?
> - Should this variable be discretized / binarized?
> - What are reasonable perturbations that could be made to this modeling pipeline? Why?


### Meeting 2

After cleaning the data and fitting models, the group will again reach out to you for your help with vetting and understanding the results. This time the questions, will center around the new analysis.

> **Examples**
>
> - Do the important variables found by students  in the fitted rules make clinical sense?
> - Are the errors the model makes reasonable?
> - Are there any surprises about the stability of the pipeline to certain perturbations?

