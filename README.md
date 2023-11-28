# Classification, regression and survival rule induction with complex elementary conditions

This repository contains data sets and code base used for experimental evaluation of the proposed algorithm of rule induction with complex elementary conditions. 

### Running experiment

To run experiments you first need a working distribution of Python installed on your machine (version 3). Then you have to install all required dependencies using following command:
```bash
pip install -r ./requirements.txt
```
To run selected experiment first move to its directory:
```bash
cd ./src/experiment/{SELECTED_EXPERIMENT_DIR}
```
Then run following command:
```
python3 ./main.py
```
or 
```bash
python ./main.py
```
Experiment results will be produced to directory: `./src/experiment/{SELECTED_EXPERIMENT_DIR}/results/`. 

## Results and their structure

Calculated results could be found in directory: `./results`. It four subdirectories:
* classifications - results for classification problem,
* regression - results for regression problem,
* survival - results for survival problem,
* notebooks - jupyter notebooks for easy analysis of the results.

In each of this directories there are subdirectories for:
* decision_tree - results of the baseline Decision Tree Model
* min_supp_X - results of experiments with specified minimal support value.

> Minimal support was used to found frequent item sets using FP Growth algorithm to search for possible M-of-N conditions candidates.

The directory structure of the collected results is as follows:
```
results/*/
├─ dataset_1/no_discretization/
│  ├─ plain/
│  │  ├─ cv/
│  │  │  ├─ 1
│  │  │  │  ├─ rules
|  |  |  |     ├─ rules.txt <-- rules for specific CV fold
│  │  │  │  ├─ metrics.csv <-- metrics for specific CV fold
|  |  |  ...
│  │  │  ├─ 10
│  │  ├─ full_dataset
|  |  |  ├─ rules
|  |  |  |  ├─ rules.txt <-- rules trained on whole data set
|  |  |  ├─ metrics.csv <-- metrics for model trained on whole data set
│  │  ├─ cv_metrics.csv <-- concatenated metrics for all CV folds
│  │  ├─ metrics.csv <-- metrics aggregated for all CV folds
│  ├─ complex/
│  ├─ excact_M-of-N/
│  ├─ at_least_M-of-N/
|  ...
├─ dataset_N/no_discretization/
├─ metrics.csv <-- aggregated metrics for all datasets
```

The root directory contains metrics.csv file with all datasets results suitable for easy analysis. It also contains directories with the results of every evaluated dataset. Those folders contain three subdirectories for every tested variant:

* plain - original RuleKit implementations with no complex conditions
* complex - complex conditions:
    * negated conditions e.g. `Color != {red}` or `Age != <18, 35)`
    * attributes relation conditions e.g. `NumberRows < NumberCols` or `NumberRows = NumberCols`
    * internal disjunctions e.g. `Color = {red, green, black}` or `(Age = <18, 34) OR Age = <45, 60))`
* excact_M-of-N - all previously mentioned complex conditions + exact M-of-N conditions (exact 2-of-3) e.g.  `2-of-3(Degree = {Bachelor's}, Age < 35, Gender = Male)`
* at_least_M-of-N - all previously mentioned complex conditions + at least M-of-N conditions (at least 2-of-3) e.g.  `2-of-3(Degree = {Bachelor's}, Age < 35, Gender = Male)`