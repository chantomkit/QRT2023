**Can you explain the price of electricity? by QRT**

https://challengedata.ens.fr/participants/challenges/97/

## Directory structures
- `QRT2023/`: Modules for DS/ML pipelines for data processing, analytics, model training, predictions in this competition
- `notebooks/`: Jupyter Notebooks for running my whole workflow

## Modelling Strategy 1
1. Splitting dataset into 3 set, according to regions and data distributions
2. Feature selection on each set, by Correlation or LASSO; And target transformation
3. Linear models will use selected features, Tree models use all features
4. Hyperparameters optimization by CV
5. Ensembles of model predictions (individual model will first be bootstrapped)

## Modelling Strategy 2
1. AutoML, with or without feature engineering

## Scores
Spearsman Correlation

| LB Type | Score |
| -------- | ------- |
| Public LB | 0.295 |
| Private LB | 0.251 |