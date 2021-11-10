---
title: cdr vetting intro
separator: '----'
verticalSeparator: '---'
highlightTheme: atom-one-dark
typora-copy-images-to: ./assets_files
revealOptions:
    transition: 'slide'
	transitionSpeed: 'fast'
---

<h1> CDR vetting </h1>

*press esc to navigate slides*

**stat 215 final project**

[![](assets_files/GitHub-Mark-64px.png)](https://github.com/Yu-Group/rule-vetting)



## logistics

- deadline: dec 12
- meeting with clinician: once week of 12/15, once week of 12/295



# project details

## helpful links

1. [project checklist](https://github.com/Yu-Group/rule-vetting#contributing-checklist)
2. [lab writeup details](https://github.com/Yu-Group/rule-vetting/blob/master/docs/lab_writeup.md)
3. [clinical collaboration details](https://github.com/Yu-Group/rule-vetting/blob/master/docs/collaborating_for_data_scientists.md)

## understanding the problem

- outcome is already selected for you

## understanding the data

- what features might be useful?

## modeling

![rule_list](assets_files/rule_list.png)

## writeup

- checking stability and judgement-calls

## 

# data-science in python

## setting up python

- `python --version` should give 3.7 or higher (might need to type `python3`)
- easier if you install things by making a [venv](https://docs.python.org/3/tutorial/venv.html)
- you can use any editor, maybe jupyterlab or [pycharm](https://www.jetbrains.com/pycharm/)
- ![pycharm](assets_files/pycharm.png)

## installation

```bash
git clone https://github.com/Yu-Group/rule-vetting
cd rule-vetting
python setup.py sdist
pip install -e .
```



## core packages

- numpy
- pandas



# custom CDR packages

## 🔎 imodels

![Screen Shot 2021-11-03 at 11.29.24 AM](assets_files/Screen%20Shot%202021-11-03%20at%2011.29.24%20AM.png)



## usage

```python
from imodels import CorelsRuleListClassifier
model = CorelsRuleListClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)
print(model)
```



## ![](https://camo.githubusercontent.com/877680cfefd1fb7cb3c240b9aca3f4cade12972550236807b2d103ba281cb3eb/68747470733a2f2f79752d67726f75702e6769746875622e696f2f76657269646963616c2d666c6f772f6c6f676f5f76666c6f775f73747261696768742e6a7067)

package for facilitating PCS analysis, especially stability

