Personal Competition Project Codebase

**Can you explain the price of electricity? by QRT**, Private LB ends at 15 Dec 2023.

https://challengedata.ens.fr/participants/challenges/97/

## Directory structures
- `QRT2023/`: self-written library and modules for DS/ML pipelines for data processing, analytics, model training, predictions in this competition
- `notebooks/`: Jupyter Notebooks for running my whole workflow

## Modelling
1. Splitting dataset into 3 set, according to regions and data distributions
2. Feature selection on each set, by Correlation or LASSO; And target transformation
3. Linear models will use selected features, Tree models use all features
4. Hyperparameters optimization by CV
5. Ensembles of model predictions (individual model will first be bootstrapped)

(WIP) Alternative strategy is to use AutoML, with or without feature engineering

## Scores
Spearsman Correlation

| LB Type | Score | Dated | Remarks |
| -------- | ------- | ------- | ------- |
| Public LB | 0.295 | June 22, 2023 | Updated realtime |
| Private LB | 0.253 | May 25, 2023 | Updated twice a year on the 15th of December and on the 15th of June |

*Note: 0.295 is probably a lucky score, my usual submission ranges from 0.25-0.26*