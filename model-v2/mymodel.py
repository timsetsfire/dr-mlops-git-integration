import pickle
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd

from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from joblib import load
import yaml
import numpy as np

class MyModel(object):
    def __init__(self, code_dir):
        """Load the model pickle file."""
        # This supports both Python 2 and 3
        with open( code_dir + "/lgbm.joblib", "rb") as picklefile:
            try:
                self.model = load(picklefile, encoding="latin1")
            except TypeError:
                self.model = load(picklefile)
        with open(code_dir + "/ordinalEncoder.joblib", "rb") as f:
            self.oe = load(f)
        with open(code_dir + "/simpleImputerNum.joblib", "rb") as f:
            self.siNum = load(f)
        with open(code_dir + "/simpleImputerCat.joblib", "rb") as f:
            self.siCat = load(f)
        with open(code_dir + "/feature_detail.yaml", "r") as f:
            self.feature_detail = yaml.load(f, Loader=yaml.FullLoader)
            self.numeric_features = self.feature_detail["Numeric"]
            self.categorical_features = self.feature_detail["Categorical"]
            self.offset = self.feature_detail["Offset"]

    def preprocess_features(self, X):
        offset = X[self.offset].values
        x_num = X[self.numeric_features].values
        x_num = self.siNum.transform(x_num)
        x_cat = X[self.categorical_features].values
        x_cat = self.siCat.transform(x_cat)
        x_cat = self.oe.transform(x_cat)
        x = np.concatenate([x_cat, x_num], axis=1)
        return (x, offset)

    def predict(
        self, X, positive_class_label=None, negative_class_label=None, **kwargs
    ):
        """
        Predict with the pickled custom model.

        If your model is for classification, you likely want to ensure this function
        calls `predict_proba()`, whereas for regression it should use `predict()`
        """
        X, offset = self.preprocess_features(X)
        predictions = np.exp(self.model.predict(X, raw_score=True)) * offset

        return pd.DataFrame(predictions, columns = ["Predictions"])
