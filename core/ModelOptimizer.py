from abc import ABC, abstractmethod
import json
import os

from scipy.stats import spearmanr
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression,
    Ridge, 
    HuberRegressor, 
    Lasso, 
    ElasticNet, 
    TheilSenRegressor,
    RANSACRegressor,
    )
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from base_dataset import Dataset

import optuna

def metric_train(output, truth):
    return spearmanr(output, truth).correlation

def _get_model(model_name):
    if model_name == 'lgbm':
        return lgb.LGBMRegressor()
    elif model_name == 'xgb':
        return xgb.XGBRegressor()
    elif model_name == 'rf':
        return RandomForestRegressor()
    elif model_name == 'svr':
        return SVR()
    elif model_name == 'ridge':
        return Ridge()
    elif model_name == 'huber':
        return HuberRegressor()
    elif model_name == 'knn':
        return KNeighborsRegressor()
    elif model_name == 'lasso':
        return Lasso()
    elif model_name == 'elasticnet':
        return ElasticNet()
    elif model_name == 'theil':
        return TheilSenRegressor()
    elif model_name == 'ransac':
        return RANSACRegressor()
    else:
        raise ValueError('Unknown model.')

scorer_train = make_scorer(metric_train)

class OptimizerPipeline(ABC):
    # def __init__(self, dataset, model_type="both", cv:int=0):
    def __init__(self, dataset: Dataset, model_type="both", cv:int=0):
        self.cv = cv
        self.train_x = dataset.dtrain.X.copy()
        self.train_y = dataset.dtrain.y.copy()
        if not self.cv:
            self.valid_x = dataset.dvalid.X.copy()
            self.valid_y = dataset.dvalid.y.copy()

        self.model, self.param = None, None
    
    @abstractmethod
    def objective(self):
        self.model.set_params(**self.param)
        tr_X, tr_y = self.train_x.values, self.train_y.values.ravel()
        if self.cv:
            return np.mean(cross_val_score(self.model, tr_X, tr_y, scoring=scorer_train, cv=self.cv))
        else:
            self.model.fit(tr_X, tr_y)
            va_X, va_y = self.valid_x.values, self.valid_y.values.ravel()
            return metric_train(va_y, self.model.predict(va_X))

    def run(self, verbose=1, seed=10):
        sampler = optuna.samplers.TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study.optimize(self.objective, n_trials=200)
        self.trial = study.best_trial
        self.model.set_params(**self.trial.params)
        
        if verbose == 1:
            print(f"Best trial among {len(study.trials)} trials:")
            print("  Value: {}".format(self.trial.value))
        if verbose == 2:
            print("  Params: ")
            for key, value in self.trial.params.items():
                print("    {}: {}".format(key, value))
        
    def dump_best_model(self, PATH):
        model_dict = {
            'value': self.trial.value,
            'params': self.model.get_params(),
        }
        json.dump(model_dict, open(PATH, 'w'))
            
            

class lgbm_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0, seed=42):
        super().__init__(dataset, model_type, cv)
        self.model = lgb.LGBMRegressor(random_state=seed)

    def objective(self, trial):
        self.param = {
            "objective": "regression",
            "metric": "regression",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 64),
        }
        return super().objective()


class xgb_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0, seed=42):
        super().__init__(dataset, model_type, cv)
        self.model = xgb.XGBRegressor(random_state=seed)

    def objective(self, trial):
        self.param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "exact",
            "lambda": trial.suggest_float("lambda", 1e-2, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-2, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 64),
            "max_leaves": trial.suggest_int("max_leaves", 2, 64),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
        return super().objective()
        

class rf_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0, seed=42):
        super().__init__(dataset, model_type, cv)
        self.model = RandomForestRegressor(random_state=seed)

    def objective(self, trial):
        self.param = {
            "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 64, step=1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 64, step=1),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 64, step=1),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-2, 1.0, log=True),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-2, 1.0, log=True),
        }
        return super().objective()

class svr_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = SVR()

    def objective(self, trial):
        self.param = {
            "C": trial.suggest_float("C", 0.5, 10.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.05, 10.0, log=True),
        }
        return super().objective()

class ridge_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = Ridge()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()

class huber_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = HuberRegressor()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1.35, 100, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()

class knn_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = KNeighborsRegressor()

    def objective(self, trial):
        self.param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 64, step=1),
            "weights": trial.suggest_categorical("weights", ['uniform', 'distance']),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
        return super().objective()

class lasso_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = Lasso()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()

class elasticnet_optimizer(OptimizerPipeline):
    def __init__(self, dataset, model_type="both", cv:int=0):
        super().__init__(dataset, model_type, cv)
        self.model = ElasticNet()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()


class model_box:
    def __init__(self, MODELS_PATH):
        self.MODELS_PATH = MODELS_PATH
        files = os.listdir(MODELS_PATH)
        self.model_names = set([f.split("_")[0] for f in files if '.json' in f])
        self.model_types = set([f.split("_")[1].replace('.json', '') for f in files if '.json' in f])

    def to_dict(self):
        model_candidates, model_scores = {}, {}
        for mname in self.model_names:
            model_type_dict, score_type_dict = {}, {}

            for mtype in self.model_types:
                with open(f'{self.MODELS_PATH}/{mname}_{mtype}.json', 'r') as fp:
                    tmp = json.load(fp)
                    model_type_dict[mtype] = _get_model(mname).set_params(**tmp['params'])
                    score_type_dict[mtype] = tmp['value']
                    fp.close()

            model_candidates[mname] = model_type_dict
            model_scores[mname] = score_type_dict
        return model_candidates, model_scores