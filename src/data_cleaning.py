import numpy as np
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.externals import joblib
from steppy.base import BaseTransformer


class DropOneValueFrequent(BaseTransformer):
    def __init__(self, threshold, constant_value=0):
        self.threshold = threshold
        self.constant_value = constant_value
        self.selected_feature_names = []

    def fit(self, X, **kwargs):
        for column in X.columns:
            counts = X[column].value_counts()
            value_fraction = counts[self.constant_value] / len(X)
            if value_fraction < self.threshold:
                self.selected_feature_names.append(column)
        return self

    def transform(self, X, **kwargs):
        return {'X': X[self.selected_feature_names]}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.selected_feature_names = params['selected_feature_names']

    def persist(self, filepath):
        params = {'selected_feature_names': self.selected_feature_names,
                  }
        joblib.dump(params, filepath)


class VarianceThreshold(BaseTransformer):
    def __init__(self, threshold):
        self.selector = fs.VarianceThreshold(threshold=threshold)

    def fit(self, X, **kwargs):
        self.selector.fit(X)

        selected_feature_indeces = self.selector.get_support()
        self.selected_feature_names = [feature for feature, is_chosen in zip(X.columns, selected_feature_indeces)
                                       if is_chosen]
        return self

    def transform(self, X, **kwargs):
        return {'X': X[self.selected_feature_names]}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.selector = params['selector']
        self.selected_feature_names = params['selected_feature_names']

    def persist(self, filepath):
        params = {'selector': self.selector,
                  'selected_feature_names': self.selected_feature_names,
                  }
        joblib.dump(params, filepath)


class DropDuplicateColumns(BaseTransformer):
    def fit(self, X, **kwargs):
        _, index = np.unique(X.values, return_index=True, axis=1)
        self.selected_feature_names = [feature for idx, feature, in enumerate(X.columns) if idx in index]
        return self

    def transform(self, X, **kwargs):
        return {'X': X[self.selected_feature_names]}

    def load(self, filepath):
        self.selected_feature_names = joblib.load(filepath)

    def persist(self, filepath):
        joblib.dump(self.selected_feature_names, filepath)


class DummiesMissing(BaseTransformer):
    def __init__(self, missing_value=0):
        self.missing_value = missing_value

    def transform(self, X, **kwargs):
        missing_mask = np.where(X.values == self.missing_value, True, False)
        missing_columns = ['{}_is_missing'.format(col) for col in X.columns]
        X_is_missing = pd.DataFrame(missing_mask.astype(int), columns=missing_columns)

        return {'categorical_features': X_is_missing}
