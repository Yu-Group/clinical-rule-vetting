*Some instructions for data-scientists on collaborating with clinicians for rule-vetting. See the [contributing checklist](https://github.com/Yu-Group/rule-vetting#contributing-checklist) for all deliverables and the [lab writeup](https://github.com/Yu-Group/rule-vetting/blob/master/docs/lab_writeup.md) for guidance.*

# Working with clinical experts

In the course of this project, you will meet twice (once in the week of 11/15 and once in the week of 11/29) with a clinical expert for domain expertise in the problem. Please take the initiative to schedule these meetings.

### Meeting 1

When cleaning the data, you will need to translate from messy data to meaningful features.
You will consult with a clinical expert to help decide which variables to use + how to process them.
Please come prepared to this meeting with a list of variables and questions about them.
Be sure to provide as much data as possible to give the medical expert the most context (e.g. what was the exact wording used when collecting a variable?).
Sharing your questions ahead of time via e-mail can help make the meeting more productive.

> **Examples**
>
> - Are these variables available at the time when the clinical decision rule is used?
> - Are the distributions of each variable reasonable?
> - Is this variable always available?
> - What does it mean for these variables to be "missing" - can they be imputed using other information?
> - Should this variable be discretized / binarized?
> - What are reasonable perturbations that could be made to this modeling pipeline?

Eventually, the final variables you use should be written into the file `data_dictionary.md`.

### Meeting 2

After cleaning the data and fitting models, clinical experts can again help vet and understand the results. Meet again and ask questions about the results of your findings.

> **Examples**
>
> - Do the important variables in the fitted rules make clinical sense?
> - Are the errors the model makes reasonable?
> - Are there any surprises about the stability of the pipeline to certain perturbations?

The analysis provided by the clinical expert should be incorporated into your final writeup.

