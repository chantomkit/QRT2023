from abc import ABC, abstractmethod
import json
import os

from scipy.stats import spearmanr
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, HuberRegressor

import optuna

def metric_train(output, truth):
    return spearmanr(output, truth).correlation

class OptimizerPipeline(ABC):
    def __init__(self, train_x, valid_x, train_y, valid_y):
        self.train_x = train_x
        self.valid_x = valid_x
        self.train_y = train_y
        self.valid_y = valid_y
        self.model, self.param = None, None
    
    @abstractmethod
    def objective(self):
        self.model.set_params(**self.param)
        self.model.fit(self.train_x, self.train_y)
        return metric_train(self.valid_y, self.model.predict(self.valid_x))

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
        with open(PATH, 'w') as fp:
            json.dump(model_dict, fp)
            
            

class lgbm_optimizer(OptimizerPipeline):
    def __init__(self, train_x, valid_x, train_y, valid_y, seed=42):
        super().__init__(train_x, valid_x, train_y, valid_y)
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
    def __init__(self, train_x, valid_x, train_y, valid_y, seed=42):
        super().__init__(train_x, valid_x, train_y, valid_y)
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
    def __init__(self, train_x, valid_x, train_y, valid_y, seed=42):
        super().__init__(train_x, valid_x, train_y, valid_y)
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
    def __init__(self, train_x, valid_x, train_y, valid_y):
        super().__init__(train_x, valid_x, train_y, valid_y)
        self.model = SVR()

    def objective(self, trial):
        self.param = {
            "C": trial.suggest_float("C", 0.5, 10.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.05, 10.0, log=True),
        }
        return super().objective()

class ridge_optimizer(OptimizerPipeline):
    def __init__(self, train_x, valid_x, train_y, valid_y):
        super().__init__(train_x, valid_x, train_y, valid_y)
        self.model = Ridge()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()

class huber_optimizer(OptimizerPipeline):
    def __init__(self, train_x, valid_x, train_y, valid_y):
        super().__init__(train_x, valid_x, train_y, valid_y)
        self.model = HuberRegressor()

    def objective(self, trial):
        self.param = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1.35, 100, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return super().objective()


class model_box:
    def __init__(self, MODELS_PATH):
        self.MODELS_PATH = MODELS_PATH
        files = os.listdir(MODELS_PATH)
        self.model_names = np.unique([f.split("_")[0] for f in files])
        self.model_types = np.unique([f.split("_")[1].strip('.json') for f in files])

    def get_model(self, model_name):
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
        else:
            raise ValueError('Unknown model.')

    def to_dicts(self):
        model_candidates, model_scores = {}, {}
        for mname in self.model_names:
            model_type_dict, score_type_dict = {}, {}

            for mtype in self.model_types:
                with open(f'{self.MODELS_PATH}/{mname}_{mtype}.json', 'r') as fp:
                    tmp = json.load(fp)
                    model_type_dict[mtype] = self.get_model(mname).set_params(**tmp['params'])
                    score_type_dict[mtype] = tmp['value']
                    fp.close()

            model_candidates[mname] = model_type_dict
            model_scores[mname] = score_type_dict
        return model_candidates, model_scores