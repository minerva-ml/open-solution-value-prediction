import os

from attrdict import AttrDict
from deepsense import neptune

from .utils import read_params, parameter_eval

ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 1000

ID_COLUMN = ['ID']
TARGET_COLUMN = ['target']

SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },

    'light_gbm': {'model_config': {'device': parameter_eval(params.lgbm__device),
                                   'boosting_type': parameter_eval(params.lgbm__boosting_type),
                                   'objective': parameter_eval(params.lgbm__objective),
                                   'metric': parameter_eval(params.lgbm__metric),
                                   'learning_rate': parameter_eval(params.lgbm__learning_rate),
                                   'max_depth': parameter_eval(params.lgbm__max_depth),
                                   'subsample': parameter_eval(params.lgbm__subsample),
                                   'colsample_bytree': parameter_eval(params.lgbm__colsample_bytree),
                                   'min_child_weight': parameter_eval(params.lgbm__min_child_weight),
                                   'reg_lambda': parameter_eval(params.lgbm__reg_lambda),
                                   'reg_alpha': parameter_eval(params.lgbm__reg_alpha),
                                   'subsample_freq': parameter_eval(params.lgbm__subsample_freq),
                                   'max_bin': parameter_eval(params.lgbm__max_bin),
                                   'min_child_samples': parameter_eval(params.lgbm__min_child_samples),
                                   'num_leaves': parameter_eval(params.lgbm__num_leaves),
                                   'nthread': parameter_eval(params.num_workers),
                                   'number_boosting_rounds': parameter_eval(params.lgbm__number_boosting_rounds),
                                   'early_stopping_rounds': parameter_eval(params.lgbm__early_stopping_rounds),
                                   'verbose': parameter_eval(params.verbose),
                                   },
                  'callback_config': {'run_with_callback': params.lgbm_random_search_runs == 0,
                                      'neptune_monitor': {'channel_prefix': 'light_gbm'}
                                      }
                  },

    'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                    'callbacks':
                                        {'neptune_monitor': {'name': 'light_gbm'},
                                         'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                      'random_search_light_gbm.pkl')}
                                         },
                                    },
                      },

})
