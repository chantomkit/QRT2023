from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

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
    def aggregate_predictions(self):
        self.predict_from_base_predictors()
        pass

class RegionalBasedPredictors(BasePredictors):
    def __init__(self, model_name:str, predictors:dict, train_data:dict, prediction_X:dict, train=True) -> None:
        super().__init__(model_name, predictors, train_data, prediction_X, train)
        for key in self.predictors.keys():
            if key not in ['fr', 'de', 'both']:
                raise ValueError('Only support fr, de or both base models')

    def aggregate_predictions(self):
        super().aggregate_predictions()
        self.predictions['fr'].columns = [f'{self.model_name}_regional_pred']
        self.predictions['de'].columns = [f'{self.model_name}_regional_pred']
        regional_pred = pd.concat([self.predictions['fr'], self.predictions['de']], axis=0)
        self.predictions['both'].columns = [f'{self.model_name}_both_pred']
        return self.predictions['both'].join(regional_pred)

class MetaPredictor(ABC):
    pass
