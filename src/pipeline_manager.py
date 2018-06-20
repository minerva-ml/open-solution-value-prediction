import os
import shutil

import pandas as pd
from deepsense import neptune
from sklearn.model_selection import train_test_split

from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .utils import init_logger, read_params, set_seed, create_submission, verify_submission, log_root_mean_squared_error

set_seed(cfg.RANDOM_SEED)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')


class PipelineManager():
    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, ):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode, submit_predictions):
        predict(pipeline_name, dev_mode, submit_predictions)


def train(pipeline_name, dev_mode):
    logger.info('TRAINING')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        application_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        application_train = pd.read_csv(params.train_filepath)

    logger.info('Shuffling and splitting into train and test...')
    train_data_split, valid_data_split = train_test_split(application_train,
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

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    logger.info('reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        application_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        application_train = pd.read_csv(params.train_filepath)

    logger.info('Shuffling and splitting to get validation split...')
    _, valid_data_split = train_test_split(application_train,
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

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(data)
    pipeline.clean_cache()

    y_pred = output['prediction']

    logger.info('Calculating LRMSE on validation set')
    score = log_root_mean_squared_error(y_true, y_pred)
    logger.info('LRMSE score on validation is {}'.format(score))
    ctx.channel_send('LRMSE', 0, score)


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

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
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
            logger.info('making Kaggle submit...')
            os.system('kaggle competitions submit -c santander-value-prediction-challenge -f {} -m {}'
                      .format(submission_filepath, params.kaggle_message))
