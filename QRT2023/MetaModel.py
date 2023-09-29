from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from QRT2023.Dataset import DataUnit

class PredictionAggregator:
    def __init__(self, models:dict, region:str):
        self.models = models
        self.region = region

    def fit_all(self, training_data:DataUnit):
        self.fitted_models = {}
        for key in self.models.keys():
            model = self.models[key][self.region]
            X, y = training_data.X.values, training_data.y.values.ravel()
            model.fit(X, y)
            self.fitted_models[key] = model

    def predict_all(self, predicting_X:pd.DataFrame):
        predictions = {}
        for key in self.models.keys():
            pred_X = predicting_X.values
            predictions[key] = self.fitted_models[key].predict(pred_X)
        return pd.DataFrame(predictions, index=predicting_X.index)

    def fit_predict(
        self, 
        training_data:DataUnit, 
        predicting_X:pd.DataFrame, 
        n_bootstrap:int=0, 
        bootstrap_fraction:float=0, 
        seed:int=42
        ):
        if n_bootstrap == 0:
            self.fit_all(training_data)
            self.full_predictions = self.predict_all(predicting_X)
        else:
            np.random.seed(seed)
            data_idx = training_data.X.index.to_list()
            bootstap_idxs = np.random.choice(
                data_idx, 
                (n_bootstrap, int(len(data_idx)*bootstrap_fraction))
                )
            bootstrap_preds = []
            for subset_idx in bootstap_idxs:
                subset = DataUnit(
                    dunit=training_data.dunit.loc[subset_idx],
                    X_cols=training_data.X_cols,
                    y_col=training_data.y_col
                    )
                self.fit_all(subset)
                bootstrap_preds.append(self.predict_all(predicting_X))
            self.full_predictions = sum(bootstrap_preds) / n_bootstrap
        return self.full_predictions