import numpy as np
import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class InferredTypeSplitter(BaseTransformer):
    def transform(self, X, **kwargs):
        numerical_columns, categorical_columns = self._get_column_types(X)

        outputs = {'numerical_features': X[numerical_columns],
                   'categorical_features': X[categorical_columns]
                   }
        return outputs

    def _get_column_types(self, X):
        types = X.dtypes.to_frame().reset_index()
        types.columns = ['colname', 'type']
        types['filter'] = types['type'].apply(self._infer_type)

        categorical_columns = types[types['filter'] == 'categorical']['colname'].tolist()
        numerical_columns = types[types['filter'] == 'numerical']['colname'].tolist()
        return numerical_columns, categorical_columns

    def _infer_type(self, x):
        x_ = str(x)
        if 'float' in x_:
            return 'numerical'
        elif 'int' in x_:
            return 'categorical'
        else:
            return 'other'


class FeatureJoiner(BaseTransformer):
    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        outputs = dict()
        outputs['features'] = pd.concat(features, axis=1).astype(np.float32)
        outputs['feature_names'] = self._get_feature_names(features)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names
