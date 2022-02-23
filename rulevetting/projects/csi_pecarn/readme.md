# csi_pecarn

**Data download 🔗**: https://pecarn.org/datasets/ "Predicting Cervical Spine Injury (CSI) in Children: A Multi-Centered Case-Control Analysis"

**Paper links 📄**: 
- primary: https://pubmed.ncbi.nlm.nih.gov/21035905/
- secondary: https://pubmed.ncbi.nlm.nih.gov/22531194/

## Replication Note
The curated set of data models, including `StableLinear`, are from the newest working version of imodels (1.2.1), which requires installing from the imodels repository
```
# Assuming in [$ROOT] which is the directory storing rule-vetting 
git clone https://github.com/csinva/imodels.git
cd imodels
pip install -e .
```
The model validation results used in `validation_results.ipynb` require results from `imodels-experiments` [repo](https://github.com/Yu-Group/imodels-experiments) and please contact authors for additional details or files required to reload the models.

To perform veridical flow analysis, we used the development distribution installed from the `vflow` repo. Specifically, we would need to delete previous `vflow` installation from `rule-env` with the following commands:
```
# in rule-env environment
cd [$ROOT]
git clone https://github.com/Yu-Group/veridical-flow.git
cd veridical-flow
python setup.py develop
# Deleting previous vflow
rm -r [$ROOT]/rule-env/lib/python3.8/site-packages/vflow
```
**Since the develop version of vflow has different function signatures, usages of `vflow` packages are modified in `rulevetting.template.dataset.py`**
