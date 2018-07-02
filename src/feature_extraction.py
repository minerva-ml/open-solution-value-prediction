import numpy as np
import pandas as pd
import sklearn.decomposition as sk_d
import sklearn.random_projection as sk_rp
from scipy.stats import skew, kurtosis
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


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


class BaseDecomposition(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = None

    def fit(self, features):
        self.estimator.fit(features)
        return self

    def transform(self, features):
        return {'features': self.estimator.transform(features)}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)


class PCA(BaseDecomposition):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = sk_d.PCA(**kwargs)


class FastICA(BaseDecomposition):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = sk_d.FastICA(**kwargs)


class FactorAnalysis(BaseDecomposition):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = sk_d.FactorAnalysis(**kwargs)


class GaussianRandomProjection(BaseDecomposition):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = sk_rp.GaussianRandomProjection(**kwargs)


class SparseRandomProjection(BaseDecomposition):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = sk_rp.SparseRandomProjection(**kwargs)


class RowAggregationFeatures(BaseTransformer):
    def __init__(self, bucket_nr, **kwargs):
        super().__init__()
        self.bucket_nr = bucket_nr

    def transform(self, X, **kwargs):
        X_aggs = []
        for i, column_bucket in enumerate(self._column_bucket_gen(X.columns)):
            X_bucket_agg = X[column_bucket].apply(aggregate_row, axis=1)
            X_bucket_agg.columns = self._add_prefix(X_bucket_agg.columns, i)
            X_aggs.append(X_bucket_agg)
        X_aggs = pd.concat(X_aggs, axis=1)
        return {'numerical_features': X_aggs}

    def _column_bucket_gen(self, cols):
        chunk_size = len(cols) // self.bucket_nr + 1
        for i in range(0, len(cols), chunk_size):
            yield cols[i:i + chunk_size]

    def _add_prefix(self, columns, bucket_id):
        columns = ['{}_of_{}_{}'.format(self.bucket_nr, bucket_id, col)
                   for col in columns]
        return columns


def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()].astype(np.float)
    if non_zero_values.empty:
        aggregations = {'non_zero_mean': np.nan,
                        'non_zero_std': np.nan,
                        'non_zero_max': np.nan,
                        'non_zero_min': np.nan,
                        'non_zero_sum': np.nan,
                        'non_zero_skewness': np.nan,
                        'non_zero_kurtosis': np.nan,
                        'non_zero_median': np.nan,
                        'non_zero_q1': np.nan,
                        'non_zero_q3': np.nan,
                        'non_zero_log_mean': np.nan,
                        'non_zero_log_std': np.nan,
                        'non_zero_log_max': np.nan,
                        'non_zero_log_min': np.nan,
                        'non_zero_log_sum': np.nan,
                        'non_zero_log_skewness': np.nan,
                        'non_zero_log_kurtosis': np.nan,
                        'non_zero_log_median': np.nan,
                        'non_zero_log_q1': np.nan,
                        'non_zero_log_q3': np.nan,
                        'non_zero_count': np.nan,
                        'non_zero_fraction': np.nan
                        }
    else:
        aggregations = {'non_zero_mean': non_zero_values.mean(),
                        'non_zero_std': non_zero_values.std(),
                        'non_zero_max': non_zero_values.max(),
                        'non_zero_min': non_zero_values.min(),
                        'non_zero_sum': non_zero_values.sum(),
                        'non_zero_skewness': skew(non_zero_values),
                        'non_zero_kurtosis': kurtosis(non_zero_values),
                        'non_zero_median': non_zero_values.median(),
                        'non_zero_q1': np.percentile(non_zero_values, q=25),
                        'non_zero_q3': np.percentile(non_zero_values, q=75),
                        'non_zero_log_mean': np.log1p(non_zero_values).mean(),
                        'non_zero_log_std': np.log1p(non_zero_values).std(),
                        'non_zero_log_max': np.log1p(non_zero_values).max(),
                        'non_zero_log_min': np.log1p(non_zero_values).min(),
                        'non_zero_log_sum': np.log1p(non_zero_values).sum(),
                        'non_zero_log_skewness': skew(np.log1p(non_zero_values)),
                        'non_zero_log_kurtosis': kurtosis(np.log1p(non_zero_values)),
                        'non_zero_log_median': np.log1p(non_zero_values).median(),
                        'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
                        'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75),
                        'non_zero_count': non_zero_values.count(),
                        'non_zero_fraction': non_zero_values.count() / row.count()
                        }
    return pd.Series(aggregations)
