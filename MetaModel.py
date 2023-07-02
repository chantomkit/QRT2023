from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class BasePredictors(ABC):
    def __init__(self, model_name:str, predictors:dict, train_data:dict, prediction_X:dict, train=True):
        self.model_name = model_name
        self.predictors = predictors
        self.train_data = train_data
        self.prediction_X = prediction_X
        self.predictions = {}
        if train:
            self.train_base_predictors()

    def train_base_predictors(self):
        for key in self.predictors.keys():
            self.predictors[key].fit(self.train_data[key]['X'], self.train_data[key]['y'])

    def predict_from_base_predictors(self, normalize=True):
        for key in self.predictors.keys():
            pred = self.predictors[key].predict(self.prediction_X[key])
            if normalize:
                pred = (pred - np.mean(pred)) / np.std(pred)
            self.predictions[key] = pd.DataFrame(pred, index=self.prediction_X[key].index)

    @abstractmethod
    def aggregate_predictions(self, normalize=True, dist_params:dict=None):
        self.predict_from_base_predictors(normalize)
        if dist_params is not None:
            for key in dist_params.keys():
                self.predictions[key] = self.predictions[key] * dist_params[key]['std'] + dist_params[key]['mean'] 

class BasicPredictors(BasePredictors):
    def __init__(self, model_name:str, predictors:dict, train_data:dict, prediction_X:dict, train=True) -> None:
        super().__init__(model_name, predictors, train_data, prediction_X, train)
        for key in self.predictors.keys():
            if key not in ['fr', 'de', 'both']:
                raise ValueError('Only support fr, de or both base models')

    def aggregate_predictions(self, normalize=True, dist_params:dict=None):
        super().aggregate_predictions(normalize=normalize, dist_params=dist_params)
        for key in self.predictions.keys():
            self.predictions[key].columns = [f'{self.model_name}_pred']
        return self.predictions

class RegionalBasedPredictors(BasePredictors):
    def __init__(self, model_name:str, predictors:dict, train_data:dict, prediction_X:dict, train=True) -> None:
        super().__init__(model_name, predictors, train_data, prediction_X, train)
        for key in self.predictors.keys():
            if key not in ['fr', 'de', 'both']:
                raise ValueError('Only support fr, de or both base models')

    def aggregate_predictions(self, normalize=True, dist_params:dict=None):
        super().aggregate_predictions(normalize=normalize, dist_params=dist_params)

        self.predictions['fr'].columns = [f'{self.model_name}_regional_pred']
        self.predictions['de'].columns = [f'{self.model_name}_regional_pred']
        regional_pred = pd.concat([self.predictions['fr'], self.predictions['de']], axis=0)
        self.predictions['both'].columns = [f'{self.model_name}_both_pred']

        return self.predictions['both'].join(regional_pred)

class StackingPredictor(BasePredictors):
    def __init__(self, model_name:str, predictors:dict, train_data:dict, prediction_X:dict, cv=5) -> None:
        super().__init__(model_name, predictors, train_data, prediction_X, train=False)
        self.cv = cv
        self.train_predictions = {}
        for key in self.predictors.keys():
            if key not in ['fr', 'de', 'both']:
                raise ValueError('Only support fr, de or both base models')
    
    def aggregate_predictions(self, normalize=True, dist_params:dict=None):
        kf = KFold(n_splits=self.cv)
        for key in self.predictors.keys():
            test_preds, preds = [], []
            for train_index, test_index in kf.split(self.train_data[key]['X']):
                X_train_tmp = self.train_data[key]['X'].iloc[train_index]
                y_train_tmp = self.train_data[key]['y'].iloc[train_index]
                X_test_tmp = self.train_data[key]['X'].iloc[test_index]
                # y_test_tmp = self.train_data[key]['y'].iloc[test_index]

                self.predictors[key].fit(X_train_tmp, y_train_tmp)
                test_pred = self.predictors[key].predict(X_test_tmp)
                pred = self.predictors[key].predict(self.prediction_X[key])

                if normalize:
                    test_pred = (test_pred - np.mean(test_pred)) / np.std(test_pred)
                    pred = (pred - np.mean(pred)) / np.std(pred)

                test_preds.append(pd.DataFrame(test_pred, index=X_test_tmp.index))
                preds.append(pred)

            self.train_predictions[key] = pd.concat(test_preds)
            self.predictions[key] = pd.DataFrame(np.array(preds).mean(axis=0), index=self.prediction_X[key].index)

        self.train_predictions['fr'].columns = [f'{self.model_name}_regional_pred']
        self.train_predictions['de'].columns = [f'{self.model_name}_regional_pred']
        regional_test_pred = pd.concat([self.train_predictions['fr'], self.train_predictions['de']], axis=0)
        self.train_predictions['both'].columns = [f'{self.model_name}_both_pred']

        first_layer_test_pred = self.train_predictions['both'].join(regional_test_pred)

        self.predictions['fr'].columns = [f'{self.model_name}_regional_pred']
        self.predictions['de'].columns = [f'{self.model_name}_regional_pred']
        regional_pred = pd.concat([self.predictions['fr'], self.predictions['de']], axis=0)
        self.predictions['both'].columns = [f'{self.model_name}_both_pred']

        first_layer_pred = self.predictions['both'].join(regional_pred)

        return first_layer_test_pred, first_layer_pred