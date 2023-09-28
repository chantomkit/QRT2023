Personal Competition Project Codebase

**Can you explain the price of electricity? by QRT**

https://challengedata.ens.fr/participants/challenges/97/

## Directory structures
- `core/`: core modules for my DS/ML pipelines for data processing, analytics, model training, predictions
- `notebooks/`: Jupyter Notebooks for running my whole workflow

## Modelling
1. Splitting dataset into 3 set, according regions and data distributions
2. Feature selection on each set, by Correlation or LASSO
3. Linear models will use selected features, Tree models use all features
4. Hyperparameters optimization by CV
5. Ensembles of model predictions (individual model predictions will first be bootstrapped)

## Scores

Spearsman Correlation

| LB Type | Score | Dated | Remarks |
| -------- | ------- | ------- | ------- |
| Public Leaderboard | 0.295 | June 22, 2023 | Updated realtime |
| Private Leaderboard | 0.253 | May 25, 2023 | Updated twice a year on the 15th of December and on the 15th of June |

*Note: 0.295 is probably a lucky score, my usual submission ranges from 0.25-0.26*