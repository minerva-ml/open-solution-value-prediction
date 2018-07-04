import lightgbm as lgb
import numpy as np
import pandas as pd
from attrdict import AttrDict
from sklearn.externals import joblib
from steppy.base import BaseTransformer

from .utils import NeptuneContext, get_logger

neptune_ctx = NeptuneContext()
logger = get_logger()


class LightGBM(BaseTransformer):
    def __init__(self, name=None, **params):
        super().__init__()
        self.msg_prefix = 'LightGBM transformer'
        logger.info('initializing {}.'.format(self.msg_prefix))
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None
        self.callbacks = callbacks(channel_prefix=name)

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
            X, y,
            X_valid, y_valid,
            feature_names='auto',
            categorical_features='auto',
            **kwargs):
        evaluation_results = {}

        self._check_target_shape_and_type(y, 'y')
        self._check_target_shape_and_type(y_valid, 'y_valid')
        y = self._format_target(y, 'y')
        y_valid = self._format_target(y_valid, 'y_valid')

        logger.info('{}, train data shape        {}'.format(self.msg_prefix, X.shape))
        logger.info('{}, validation data shape   {}'.format(self.msg_prefix, X_valid.shape))
        logger.info('{}, train labels shape      {}'.format(self.msg_prefix, y.shape))
        logger.info('{}, validation labels shape {}'.format(self.msg_prefix, y_valid.shape))

        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)
        data_valid = lgb.Dataset(X_valid,
                                 label=y_valid,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)

        self.estimator = lgb.train(self.model_config,
                                   data_train,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   valid_sets=[data_train, data_valid],
                                   valid_names=['data_train', 'data_valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function,
                                   callbacks=self.callbacks,
                                   **kwargs)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def _check_target_shape_and_type(self, target, name):
        if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            raise TypeError(
                '{}: "{}" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(
                    self.msg_prefix,
                    name,
                    type(target)))
        try:
            assert len(target.shape) == 1, '{}: "{}" must be 1-D. It is {}-D instead.'.format(self.msg_prefix,
                                                                                              name,
                                                                                              len(target.shape))
        except AttributeError:
            print('{}: cannot determine shape of the {}.'
                  'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(self.msg_prefix,
                                                                                                     name,
                                                                                                     type(target)))

    def _format_target(self, target, name):
        if isinstance(target, pd.Series):
            return target.values
        elif isinstance(target, np.ndarray):
            return target
        elif isinstance(target, list):
            return np.array(target)
        else:
            raise TypeError('{}: "{}" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(
                self.msg_prefix,
                name,
                type(target)))


def callbacks(channel_prefix):
    neptune_monitor = neptune_monitor_lgbm(channel_prefix)
    return [neptune_monitor]


def neptune_monitor_lgbm(channel_prefix=''):
    def callback(env):
        for name, loss_name, loss_value, _ in env.evaluation_result_list:
            if channel_prefix != '':
                channel_name = '{}_{}_{}'.format(channel_prefix, name, loss_name)
            else:
                channel_name = '{}_{}'.format(name, loss_name)
            neptune_ctx.ctx.channel_send(channel_name, x=env.iteration, y=loss_value)
    return callback
