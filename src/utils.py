import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from deepsense import neptune
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from steppy.base import BaseTransformer


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class NeptuneContext(_Borg):
    def __init__(self, fallback_file='neptune_local.yaml'):
        _Borg.__init__(self)

        self.ctx = neptune.Context()
        self.fallback_file = fallback_file
        self.params = self._read_params()
        self.numeric_channel = neptune.ChannelType.NUMERIC
        self.image_channel = neptune.ChannelType.IMAGE
        self.text_channel = neptune.ChannelType.TEXT

    def _read_params(self):
        if self.ctx.params.__class__.__name__ == 'OfflineContextParams':
            params = self._read_yaml().parameters
        else:
            params = self.ctx.params
        return params

    def _read_yaml(self):
        with open(self.fallback_file) as f:
            config = yaml.load(f)
        return AttrDict(config)


def parameter_eval(param):
    try:
        return eval(param)
    except Exception:
        return param


def create_submission(meta, predictions):
    submission = pd.DataFrame({'ID': meta['ID'].tolist(),
                               'target': predictions
                               })
    return submission


def verify_submission(submission, sample_submission):
    assert submission.shape == sample_submission.shape, \
        'Expected submission to have shape {} but got {}'.format(sample_submission.shape, submission.shape)

    for submission_id, correct_id in zip(submission['ID'].values, sample_submission['ID'].values):
        assert correct_id == submission_id, \
            'Wrong id: expected {} but got {}'.format(correct_id, submission_id)


def get_logger():
    return logging.getLogger('value-prediction')


def init_logger():
    logger = logging.getLogger('value-prediction')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def persist_evaluation_predictions(experiment_directory, y_pred, raw_data, id_column, target_column):
    raw_data.loc[:, 'y_pred'] = y_pred.reshape(-1)
    predictions_df = raw_data.loc[:, [id_column, target_column, 'y_pred']]
    filepath = os.path.join(experiment_directory, 'evaluation_predictions.csv')
    logging.info('evaluation predictions csv shape: {}'.format(predictions_df.shape))
    predictions_df.to_csv(filepath, index=None)


def set_seed(seed=90210):
    random.seed(seed)
    np.random.seed(seed)


def make_transformer(func, output_name):
    class StaticTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            return {output_name: func(*args, **kwargs)}

    return StaticTransformer()


def to_pandas(x, column_prefix):
    col_num = x.shape[1]
    df = pd.DataFrame(x, columns=['{}_{}'.format(column_prefix, i) for i in range(col_num)])
    return df


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)))


class KFoldByTargetValue(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split