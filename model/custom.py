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

    def _determine_positive_class_index(self, positive_class_label):
        """Find index of positive class label to interpret predict_proba output"""
        labels = [str(label) for label in self.model.classes_]
        try:
            return labels.index(positive_class_label)
        except ValueError:
            return 1

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

"""
Custom hooks for prediction
---------------------------

If drum's standard assumptions are incorrect for your model, **drum** supports several hooks 
for custom inference code. 
"""


def init(code_dir: Optional[str], **kwargs) -> None:
    """

    Parameters
    ----------
    code_dir : code folder passed in the `--code_dir` parameter
    kwargs : future proofing
    """


def load_model(code_dir: str) -> Any:
    """
    Can be used to load supported models if your model has multiple artifacts, or for loading
    models that **drum** does not natively support

    Parameters
    ----------
    code_dir : is the directory where model artifact and additional code are provided, passed in

    Returns
    -------
    If used, this hook must return a non-None value
    """

    return MyModel(code_dir)


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if **drum** supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to **drum** to make predictions on
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied

    Returns
    -------
    Transformed data
    """
    return data


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    This hook is only needed if you would like to use **drum** with a framework not natively
    supported by the tool.

    Parameters
    ----------
    data : is the dataframe to make predictions against. If `transform` is supplied,
    `data` will be the transformed data.
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
    kwargs : additional keyword arguments to the method
    In case of classification model class labels will be provided as the following arguments:
    - `positive_class_label` is the positive class label for a binary classification model
    - `negative_class_label` is the negative class label for a binary classification model

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Binary Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row should sum to 1.0
      Regression: must have a single column called `Predictions` with numerical values

    """
    return model.predict(data)


def post_process(predictions: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    This method is only needed if your model's output does not match the above expectations

    Parameters
    ----------
    predictions : is the dataframe of predictions produced by **drum** or by
      the `score` hook, if supplied
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied

    Returns
    -------
    This method should return predictions as a dataframe with the following format:
      Binary Classification: must have columns for each class label with floating- point class
        probabilities as values. Each row
    should sum to 1.0
      Regression: must have a single column called `Predictions` with numerical values

    """
    return predictions
