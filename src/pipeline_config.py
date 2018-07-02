import os

from attrdict import AttrDict

from .utils import NeptuneContext, parameter_eval

neptune_ctx = NeptuneContext()
params = neptune_ctx.params

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 500

ID_COLUMN = ['ID']
TARGET_COLUMN = ['target']
MISSING_VALUE = 0

SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },

    'variance_threshold': {'threshold': params.variance_threshold__threshold
                           },
    'drop_zero_fraction': {'threshold': params.drop_zero_fraction__threshold
                           },
    'dummies_missing': {'missing_value': MISSING_VALUE,
                        },

    'row_aggregations': {'bucket_nrs': parameter_eval(params.row_aggregations__bucket_nrs)
                         },

    'truncated_svd': {'use': params.truncated_svd__use,
                      'params': {'n_components': params.truncated_svd__n_components,
                                 'n_iter': params.truncated_svd__n_iter,
                                 'random_state': RANDOM_SEED
                                 }
                      },
    'pca': {'use': params.pca__use,
            'params': {'n_components': params.pca__n_components,
                       'random_state': RANDOM_SEED
                       }
            },
    'fast_ica': {'use': params.fast_ica__use,
                 'params': {
                     'n_components': params.fast_ica__n_components,
                     'random_state': RANDOM_SEED
                 }
                 },
    'factor_analysis': {'use': params.factor_analysis__use,
                        'params': {
                            'n_components': params.factor_analysis__n_components,
                            'random_state': RANDOM_SEED
                        }
                        },
    'gaussian_random_projection': {'use': params.gaussian_random_projection__use,
                                   'params': {
                                       'n_components': params.gaussian_random_projection__n_components,
                                       'eps': params.gaussian_projection__eps,
                                       'random_state': RANDOM_SEED
                                   }
                                   },
    'sparse_random_projection': {'use': params.sparse_random_projection__use,
                                 'params': {
                                     'n_components': params.sparse_random_projection__n_components,
                                     'dense_output': True,
                                     'random_state': RANDOM_SEED
                                 }
                                 },

    'light_gbm': {'device': parameter_eval(params.lgbm__device),
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
                  'zero_as_missing': parameter_eval(params.lgbm__zero_as_missing),
                  'verbose': parameter_eval(params.verbose),
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
