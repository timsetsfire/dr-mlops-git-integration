
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
import io

from mymodel import MyModel

import io
from io import BytesIO

#
#"""
#This file was autogenerated by: drum new model --language python
#Generation date: Mon Nov  9 14:31:54 2020
#
#Note: this is an example of custom.py file.
#    Below are all the hooks you can use to provide your own implementation.
#    All hooks are currently commented out so uncomment a hook function in 
#    order to use it.
#"""
#
#
#def init(**kwargs):
#    """
#    This hook can be implemented to adjust logic in the training and scoring mode.
#    init is called once the code is started.
#
#    :param kwargs: additional keyword arguments to the function.
#    code_dir - code folder passed in --code_dir argument
#    """
#    pass
#
#
def load_model(input_dir):
   """
   This hook can be implemented to adjust logic in the scoring mode.

   load_model hook provides a way to implement model loading your self.
   This function should return an object that represents your model. This object will
   be passed to the predict hook for performing predictions.
   This hook can be used to load supported models if your model has multiple artifacts, or
   for loading models that drum does not natively support

   :param input_dir: the directory to load serialized models from
   :returns: Object containing the model - the predict hook will get this object as a parameter
   """

   # Returning a string with value "dummy" as the model.
   return MyModel(input_dir)
#
#
#def transform(data, model):
#    """
#    This hook can be implemented to adjust logic in the scoring mode.
#
#    transform(data: DataFrame, model: Any) -> DataFrame
#
#    Intended to apply transformations to the prediction data before making predictions.
#    This is most useful if drum supports the model's library, but your model requires additional
#    data processing before it can make predictions
#
#    :param data: dataframe given to drum to make predictions on
#    :param model: is the deserialized model loaded by drum or by load_model hook , if supplied
#    :returns: a dataframe after transformation needed
#    """
#    return data
#
#
def score(data, model, **kwargs):
   """
   This hook can be implemented to adjust logic in the scoring mode.

   This method should return predictions as a dataframe with the following format:

   Binary Classification:
   Must have columns for each class label with floating-point class probabilities as values.
   Each row should sum to 1.0

   Regression:
   Must have a single column called "Predictions" with numerical values

   This hook is only needed if you would like to use drum with a framework not natively
   supported by the tool.

   :param data: the dataframe to make predictions against. If transform is supplied, data
       will be the transformed data.
   :param model: is the deserialized model loaded by drum or by load_model hook, if supplied
   :param kwargs: additional keyword arguments to the function. If model is binary classification,
   positive_class_label and negative_class_label will be provided in kwargs. If the model is multiclass
   classification (at least 3 classes), a class_labels list will be provided as a parameter.
   :returns: a dataframe, see documentation above on the structure of the dataframe to return.
   """

   return model.predict(data)