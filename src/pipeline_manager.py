import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.model_selection import train_test_split

from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .utils import init_logger, NeptuneContext, set_seed, \
    create_submission, verify_submission, \
    root_mean_squared_log_error, KFoldByTargetValue

neptune_ctx = NeptuneContext()
params = neptune_ctx.params
ctx = neptune_ctx.ctx

set_seed(cfg.RANDOM_SEED)
logger = init_logger()


class PipelineManager:
    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, ):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode, submit_predictions):
        predict(pipeline_name, dev_mode, submit_predictions)

    def train_evaluate_cv(self, pipeline_name, dev_mode):
        train_evaluate_cv(pipeline_name, dev_mode)

    def train_evaluate_predict_cv(self, pipeline_name, dev_mode, submit_predictions):
        train_evaluate_predict_cv(pipeline_name, dev_mode, submit_predictions)


def train(pipeline_name, dev_mode):
    logger.info('TRAINING')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        train = pd.read_csv(params.train_filepath)

    logger.info('Shuffling and splitting into train and test...')
    train_data_split, valid_data_split = train_test_split(train,
                                                          test_size=params.validation_size,
                                                          random_state=cfg.RANDOM_SEED,
                                                          shuffle=params.shuffle)

    logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Train shape: {}'.format(train_data_split.shape))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    data = {'input': {'X': train_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                      'y': train_data_split[cfg.TARGET_COLUMN].values.reshape(-1),
                      'X_valid': valid_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                      'y_valid': valid_data_split[cfg.TARGET_COLUMN].values.reshape(-1),
                      },
            }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True, suffix='')
    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    logger.info('reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        train = pd.read_csv(params.train_filepath)

    logger.info('Shuffling and splitting to get validation split...')
    _, valid_data_split = train_test_split(train,
                                           test_size=params.validation_size,
                                           random_state=cfg.RANDOM_SEED,
                                           shuffle=params.shuffle)

    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    y_true = valid_data_split[cfg.TARGET_COLUMN].values
    data = {'input': {'X': valid_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False, suffix='')
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(data)
    pipeline.clean_cache()

    y_pred = output['prediction']

    logger.info('Calculating RMSLE on validation set')
    score = root_mean_squared_log_error(y_true, y_pred)
    logger.info('RMSLE score on validation is {}'.format(score))
    ctx.channel_send('RMSLE', 0, score)


def predict(pipeline_name, dev_mode, submit_predictions):
    logger.info('PREDICTION')
    logger.info('reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        test = pd.read_csv(params.test_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        test = pd.read_csv(params.test_filepath)

    data = {'input': {'X': test.drop(cfg.ID_COLUMN, axis=1),
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False, suffix='')
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['prediction']

    if not dev_mode:
        logger.info('creating submission file...')
        submission = create_submission(test, y_pred)
        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(submission, sample_submission)

        submission_filepath = os.path.join(params.experiment_directory, 'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('submission persisted to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission.head()))

        if submit_predictions and params.kaggle_api:
            make_submission(submission)


def make_submission(submission_filepath):
    logger.info('Making Kaggle submit...')
    os.system('kaggle competitions submit -c santander-value-prediction-challenge -f {} -m {}'.format(
        submission_filepath,
        params.kaggle_message))
    logger.info('Kaggle submit completed')


def train_evaluate_cv(pipeline_name, dev_mode):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')
    if dev_mode:
        logger.info('Running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        train = pd.read_csv(params.train_filepath)

    cv = KFoldByTargetValue(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
    target_values = train[cfg.TARGET_COLUMN].values.reshape(-1)

    fold_scores = []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(target_values)):
        train_data_split, valid_data_split = train.iloc[train_idx], train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMN].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, _, _ = _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id, pipeline_name)

        logger.info('Fold {} RMSLE {}'.format(fold_id, score))
        ctx.channel_send('Fold {} RMSLE'.format(fold_id), 0, score)

        fold_scores.append(score)

    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('RMSLE mean {}, RMSLE std {}'.format(score_mean, score_std))
    ctx.channel_send('RMSLE', 0, score_mean)
    ctx.channel_send('RMSLE STD', 0, score_std)


def train_evaluate_predict_cv(pipeline_name, dev_mode, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        test = pd.read_csv(params.test_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        train = pd.read_csv(params.train_filepath)
        test = pd.read_csv(params.test_filepath)

    cv = KFoldByTargetValue(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
    target_values = train[cfg.TARGET_COLUMN].values.reshape(-1)

    out_of_fold_train_predictions, out_of_fold_test_predictions, fold_scores = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(target_values)):
        train_data_split, valid_data_split = train.iloc[train_idx], train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMN].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                         valid_data_split, test,
                                                                                         fold_id, pipeline_name)

        logger.info('Fold {} RMSLE {}'.format(fold_id, score))
        ctx.channel_send('Fold {} RMSLE'.format(fold_id), 0, score)

        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)
        fold_scores.append(score)

    out_of_fold_train_predictions = pd.concat(out_of_fold_train_predictions, axis=0)
    out_of_fold_test_predictions = pd.concat(out_of_fold_test_predictions, axis=0)

    test_prediction_aggregated = _aggregate_test_prediction(out_of_fold_test_predictions)
    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('RMSLE mean {}, RMSLE std {}'.format(score_mean, score_std))
    ctx.channel_send('RMSLE', 0, score_mean)
    ctx.channel_send('RMSLE STD', 0, score_std)

    logger.info('Saving predictions')
    out_of_fold_train_predictions.to_csv(os.path.join(params.experiment_directory,
                                                      '{}_out_of_fold_train_predictions.csv'.format(pipeline_name)),
                                         index=None)
    out_of_fold_test_predictions.to_csv(os.path.join(params.experiment_directory,
                                                     '{}_out_of_fold_test_predictions.csv'.format(pipeline_name)),
                                        index=None)
    test_aggregated_file_path = os.path.join(params.experiment_directory,
                                             '{}_test_predictions_{}.csv'.format(pipeline_name,
                                                                                 params.aggregation_method))
    test_prediction_aggregated.to_csv(test_aggregated_file_path, index=None)

    if not dev_mode:
        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(test_prediction_aggregated, sample_submission)

        if submit_predictions and params.kaggle_api:
            make_submission(test_aggregated_file_path)


def _fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, test, fold_id, pipeline_name):
    score, y_valid_pred, pipeline = _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id, pipeline_name)

    test_data = {'input': {'X': test.drop(cfg.ID_COLUMN, axis=1),
                           'y': None,
                           },
                 }

    logger.info('Start pipeline transform on test')
    pipeline.clean_cache()
    output_test = pipeline.transform(test_data)
    pipeline.clean_cache()
    y_test_pred = output_test['prediction']

    train_out_of_fold_prediction_chunk = valid_data_split[cfg.ID_COLUMN]
    train_out_of_fold_prediction_chunk['fold_id'] = fold_id
    train_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_valid_pred

    test_out_of_fold_prediction_chunk = test[cfg.ID_COLUMN]
    test_out_of_fold_prediction_chunk['fold_id'] = fold_id
    test_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_test_pred

    return score, train_out_of_fold_prediction_chunk, test_out_of_fold_prediction_chunk


def _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id, pipeline_name):
    train_data = {'input': {'X': train_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                            'y': train_data_split[cfg.TARGET_COLUMN].values.reshape(-1),
                            'X_valid': valid_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                            'y_valid': valid_data_split[cfg.TARGET_COLUMN].values.reshape(-1),
                            },
                  }

    valid_data = {'input': {'X': valid_data_split.drop(cfg.TARGET_COLUMN + cfg.ID_COLUMN, axis=1),
                            'y': None,
                            },
                  }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True,
                                        suffix='_fold_{}'.format(fold_id))

    logger.info('Start pipeline fit and transform on train')
    pipeline.clean_cache()
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False,
                                        suffix='_fold_{}'.format(fold_id))
    logger.info('Start pipeline transform on valid')
    pipeline.clean_cache()
    output_valid = pipeline.transform(valid_data)
    pipeline.clean_cache()

    y_valid_pred = output_valid['prediction']
    y_valid_true = valid_data_split[cfg.TARGET_COLUMN].values
    score = root_mean_squared_log_error(y_valid_true, y_valid_pred)

    return score, y_valid_pred, pipeline


def _aggregate_test_prediction(out_of_fold_test_predictions):
    agg_methods = {'mean': np.mean,
                   'gmean': gmean}
    prediction_column = [col for col in out_of_fold_test_predictions.columns if '_prediction' in col]
    test_prediction_aggregated = out_of_fold_test_predictions.groupby(cfg.ID_COLUMN)[prediction_column].apply(
        agg_methods[params.aggregation_method]).reset_index()

    test_prediction_aggregated.columns = [cfg.ID_COLUMN + cfg.TARGET_COLUMN]

    return test_prediction_aggregated
