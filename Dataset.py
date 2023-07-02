from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

class dataset:
    def __init__(
        self, 
        dfull:pd.DataFrame=None, 
        dtrain:pd.DataFrame=None, 
        dvalid:pd.DataFrame=None, 
        dholdout:pd.DataFrame=None,
        valid_ratio:float=0.2,
        holdout_ratio:float=0,
        exclude_X_cols:list=None,
        y_col:str=None,
        split=False,
    ):
        self.dfull = dfull
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.dholdout = dholdout
        if split:
            self.data_split(valid_ratio, holdout_ratio)

        self.X_y_split(exclude_X_cols, y_col)

    def data_split(self, valid_ratio:float=0.2, holdout_ratio:float=0):
        if self.dfull is not None:
            train_end_idx = int((1 - valid_ratio - holdout_ratio) * len(self.dfull))
            valid_end_idx = int((1 - holdout_ratio) * len(self.dfull))
            self.dtrain = self.dfull.iloc[:train_end_idx]
            self.dvalid = self.dfull.iloc[train_end_idx:valid_end_idx]
            self.dholdout =  self.dfull.iloc[valid_end_idx:]
        else:
            raise ValueError("dfull is not specified.")

    def X_y_split(self, exclude_X_cols:list=None, y_col:str=None):
        self.X_full, self.y_full, self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_holdout, self.y_holdout = None, None, None, None, None, None, None, None
        if self.dfull is not None:
            if exclude_X_cols is not None: self.X_full = self.dfull.drop(exclude_X_cols, axis=1)
            if y_col is not None: self.y_full = self.dfull[y_col]

        if self.dtrain is not None:
            if exclude_X_cols is not None: self.X_train = self.dtrain.drop(exclude_X_cols, axis=1)
            if y_col is not None: self.y_train = self.dtrain[y_col]

        if self.dvalid is not None:
            if exclude_X_cols is not None: self.X_valid = self.dvalid.drop(exclude_X_cols, axis=1)
            if y_col is not None: self.y_valid = self.dvalid[y_col]
            
        if self.dholdout is not None:
            if exclude_X_cols is not None: self.X_holdout = self.dholdout.drop(exclude_X_cols, axis=1)
            if y_col is not None: self.y_holdout = self.dholdout[y_col]

class RegionalDatasets(dataset):
    def __init__(
        self, 
        dfull:pd.DataFrame, 
        dfr:pd.DataFrame=None,
        dde:pd.DataFrame=None,
        valid_ratio:float=0.2, 
        holdout_ratio:float=0, 
        exclude_X_cols_by_region:dict=None, 
        y_col:str=None,
    ):
        self.dataset_full = dataset(
            dfull=dfull, 
            valid_ratio=valid_ratio, 
            holdout_ratio=holdout_ratio, 
            exclude_X_cols=exclude_X_cols_by_region['both'], 
            y_col=y_col, 
            split=True
        )
        
        tmp_full = self.dataset_full.dfull.copy()
        tmp_train = self.dataset_full.dtrain.copy()
        tmp_valid = self.dataset_full.dvalid.copy()
        tmp_holdout = self.dataset_full.dholdout.copy()

        if dfr is None:
            self.dataset_fr = dataset(
                dfull=tmp_full.loc[tmp_full.COUNTRY == 0],
                dtrain=tmp_train.loc[tmp_train.COUNTRY == 0],
                dvalid=tmp_valid.loc[tmp_valid.COUNTRY == 0],
                dholdout=tmp_holdout.loc[tmp_holdout.COUNTRY == 0],
                exclude_X_cols=exclude_X_cols_by_region['fr'], 
                y_col=y_col
            )
        else:
            self.dataset_fr = dataset(
                dfull=dfr, 
                valid_ratio=valid_ratio, 
                holdout_ratio=holdout_ratio, 
                exclude_X_cols=exclude_X_cols_by_region['fr'], 
                y_col=y_col, 
                split=True
            )

        if dde is None:
            self.dataset_de = dataset(
                dfull=tmp_full.loc[tmp_full.COUNTRY == 1],
                dtrain=tmp_train.loc[tmp_train.COUNTRY == 1],
                dvalid=tmp_valid.loc[tmp_valid.COUNTRY == 1],
                dholdout=tmp_holdout.loc[tmp_holdout.COUNTRY == 1],
                exclude_X_cols=exclude_X_cols_by_region['de'], 
                y_col=y_col
            )
        else:
            self.dataset_de = dataset(
                dfull=dde, 
                valid_ratio=valid_ratio, 
                holdout_ratio=holdout_ratio, 
                exclude_X_cols=exclude_X_cols_by_region['de'], 
                y_col=y_col, 
                split=True
            )

    def to_dict(self):
        data_dict = {
            'full': 
            {
                'fr':
                {
                    'X': self.dataset_fr.X_full,
                    'y': self.dataset_fr.y_full,
                },
                'de':
                {
                    'X': self.dataset_de.X_full,
                    'y': self.dataset_de.y_full,
                },
                'both':
                {
                    'X': self.dataset_full.X_full,
                    'y': self.dataset_full.y_full,
                },
            },
            'train': 
            {
                'fr':
                {
                    'X': self.dataset_fr.X_train,
                    'y': self.dataset_fr.y_train,
                },
                'de':
                {
                    'X': self.dataset_de.X_train,
                    'y': self.dataset_de.y_train,
                },
                'both':
                {
                    'X': self.dataset_full.X_train,
                    'y': self.dataset_full.y_train,
                },
            },
            'valid': 
            {
                'fr':
                {
                    'X': self.dataset_fr.X_valid,
                    'y': self.dataset_fr.y_valid,
                },
                'de':
                {
                    'X': self.dataset_de.X_valid,
                    'y': self.dataset_de.y_valid,
                },
                'both':
                {
                    'X': self.dataset_full.X_valid,
                    'y': self.dataset_full.y_valid,
                },
            },
            'holdout': 
            {
                'fr':
                {
                    'X': self.dataset_fr.X_holdout,
                    'y': self.dataset_fr.y_holdout,
                },
                'de':
                {
                    'X': self.dataset_de.X_holdout,
                    'y': self.dataset_de.y_holdout,
                },
                'both':
                {
                    'X': self.dataset_full.X_holdout,
                    'y': self.dataset_full.y_holdout,
                },
            },
        }
        return data_dict

