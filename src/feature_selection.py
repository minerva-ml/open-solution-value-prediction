import numpy as np
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.linear_model import LassoCV
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger


class LassoFeatureSelection(BaseTransformer):
    def __init__(self, threshold, n_jobs, n_alphas, random_state):
        self.selector = fs.SelectFromModel(LassoCV(n_jobs=n_jobs,
                                                   n_alphas=n_alphas,
                                                   random_state=random_state),
                                           threshold=threshold)

    def fit(self, features, target, feature_names, categorical_features, **kwargs):
        self.selector.fit(features, target)

        selected_feature_indeces = self.selector.get_support()
        self.selected_feature_names = [feature for feature, is_chosen in zip(feature_names, selected_feature_indeces)
                                       if is_chosen]
        self.selected_categorical_names = [categorical_feature for categorical_feature in categorical_features
                                           if categorical_feature in self.selected_feature_names]
        return self

    def transform(self, features, **kwargs):
        return {'features': features[self.selected_feature_names],
                'feature_names': self.selected_feature_names,
                'categorical_features': self.selected_categorical_names}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.selector = params['selector']
        self.selected_feature_names = params['selected_feature_names']
        self.selected_categorical_names = params['selected_categorical_names']

    def persist(self, filepath):
        params = {'selector': self.selector,
                  'selected_feature_names': self.selected_feature_names,
                  'selected_categorical_names': self.selected_categorical_names
                  }
        joblib.dump(params, filepath)
