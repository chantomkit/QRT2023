from dataclasses import dataclass, field

from enum import Enum
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

class DataSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    HOLDOUT = "holdout"

@dataclass
class DataUnit:
    dunit: pd.DataFrame
    X_cols: List
    y_col: Optional[List]

    metadata: Dict = field(default_factory=dict)

    X: pd.DataFrame = field(init=False)
    y: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.X = self.dunit[self.X_cols]
        self.y = self.dunit[self.y_col] if self.y_col else None

@dataclass
class Dataset:
    data: pd.DataFrame
    X_cols: List
    y_col: List
    name: str = "dataset"

    valid_ratio: Optional[float] = 0.2
    holdout_ratio: Optional[float] = 0
    split_seed: Optional[int] = 42

    dfull: DataUnit = field(init=False)
    dtrain: DataUnit = field(init=False)
    dvalid: DataUnit = field(init=False)
    dholdout: DataUnit = field(init=False)

    def _apply_data_split(self):
        n = len(self.data)
        idxs = np.arange(n)
        if self.split_seed is not None:
            np.random.seed(self.split_seed)
            np.random.shuffle(idxs)

        train_end_idx = int((1 - self.valid_ratio - self.holdout_ratio) * n)
        valid_end_idx = int((1 - self.holdout_ratio) * n)

        self.dtrain = DataUnit(
            self.data.iloc[idxs[:train_end_idx]], 
            X_cols=self.X_cols,
            y_col=self.y_col,
            metadata={
                "name": self.name,
                "data_split": DataSplit.TRAIN
            },
        )

        self.dvalid = DataUnit(
            self.data.iloc[idxs[train_end_idx:valid_end_idx]], 
            X_cols=self.X_cols,
            y_col=self.y_col,
            metadata={
                "name": self.name,
                "data_split": DataSplit.VALID
            },
        ) if self.valid_ratio else None
            
        self.dholdout =  DataUnit(
            self.data.iloc[idxs[valid_end_idx:]], 
            X_cols=self.X_cols,
            y_col=self.y_col,
            metadata={
                "name": self.name,
                "data_split": DataSplit.HOLDOUT
            },
        ) if self.holdout_ratio else None

    def __post_init__(self):
        self.dfull = DataUnit(
            self.data,
            X_cols=self.X_cols,
            y_col=self.y_col,
            metadata={
                "name": self.name,
            },
        )
        self._apply_data_split()