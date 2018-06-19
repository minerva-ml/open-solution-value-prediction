import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, parameter_eval

ctx = neptune.Context()
params = read_params(ctx)

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 1000

ID_COLUMN = None
TARGET_COLUMN = None

TIMESTAMP_COLUMNS = []
CATEGORICAL_COLUMNS = []
NUMERICAL_COLUMNS = []
USELESS_COLUMNS = []

AGGREGATION_RECIPIES = []

SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },

    'preprocessing': {'fillna_value': params.fillna_value},

    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': TIMESTAMP_COLUMNS,
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
                  'verbose': parameter_eval(params.verbose),
                  },

    'xgboost': {'booster': parameter_eval(params.xgb__booster),
                'objective': parameter_eval(params.xgb__objective),
                'tree_method': parameter_eval(params.xgb__tree_method),
                'eval_metric': parameter_eval(params.xgb__eval_metric),
                'eta': parameter_eval(params.xgb__eta),
                'max_depth': parameter_eval(params.xgb__max_depth),
                'subsample': parameter_eval(params.xgb__subsample),
                'colsample_bytree': parameter_eval(params.xgb__colsample_bytree),
                'colsample_bylevel': parameter_eval(params.xgb__colsample_bylevel),
                'min_child_weight': parameter_eval(params.xgb__min_child_weight),
                'lambda': parameter_eval(params.xgb__lambda),
                'alpha': parameter_eval(params.xgb__alpha),
                'max_bin': parameter_eval(params.xgb__max_bin),
                'num_leaves': parameter_eval(params.xgb__max_leaves),
                'nthread': parameter_eval(params.num_workers),
                'nrounds': parameter_eval(params.xgb__nrounds),
                'early_stopping_rounds': parameter_eval(params.xgb__early_stopping_rounds),
                'verbose': parameter_eval(params.verbose)
                },

    'random_forest': {'n_estimators': parameter_eval(params.rf__n_estimators),
                      'criterion': parameter_eval(params.rf__criterion),
                      'max_features': parameter_eval(params.rf__max_features),
                      'min_samples_split': parameter_eval(params.rf__min_samples_split),
                      'min_samples_leaf': parameter_eval(params.rf__min_samples_leaf),
                      'n_jobs': parameter_eval(params.num_workers),
                      'random_state': RANDOM_SEED,
                      'verbose': parameter_eval(params.verbose),
                      'class_weight': parameter_eval(params.rf__class_weight),
                      },

    'logistic_regression': {'penalty': parameter_eval(params.lr__penalty),
                            'tol': parameter_eval(params.lr__tol),
                            'C': parameter_eval(params.lr__C),
                            'fit_intercept': parameter_eval(params.lr__fit_intercept),
                            'class_weight': parameter_eval(params.lr__class_weight),
                            'random_state': RANDOM_SEED,
                            'solver': parameter_eval(params.lr__solver),
                            'max_iter': parameter_eval(params.lr__max_iter),
                            'verbose': parameter_eval(params.verbose),
                            'n_jobs': parameter_eval(params.num_workers),
                            },

    'svc': {'kernel': parameter_eval(params.svc__kernel),
            'C': parameter_eval(params.svc__C),
            'degree': parameter_eval(params.svc__degree),
            'gamma': parameter_eval(params.svc__gamma),
            'coef0': parameter_eval(params.svc__coef0),
            'probability': parameter_eval(params.svc__probability),
            'tol': parameter_eval(params.svc__tol),
            'max_iter': parameter_eval(params.svc__max_iter),
            'verbose': parameter_eval(params.verbose),
            'random_state': RANDOM_SEED,
            },

    'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                    'callbacks':
                                        {'neptune_monitor': {'name': 'light_gbm'},
                                         'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                      'random_search_light_gbm.pkl')}
                                         },
                                    },
                      'xgboost': {'n_runs': params.xgb_random_search_runs,
                                  'callbacks':
                                      {'neptune_monitor': {'name': 'xgboost'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_xgboost.pkl')}
                                       },
                                  },
                      'random_forest': {'n_runs': params.rf_random_search_runs,
                                        'callbacks':
                                            {'neptune_monitor': {'name': 'random_forest'},
                                             'persist_results':
                                                 {'filepath': os.path.join(params.experiment_directory,
                                                                           'random_search_random_forest.pkl')}
                                             },
                                        },
                      'logistic_regression': {'n_runs': params.lr_random_search_runs,
                                              'callbacks':
                                                  {'neptune_monitor': {'name': 'logistic_regression'},
                                                   'persist_results':
                                                       {'filepath': os.path.join(params.experiment_directory,
                                                                                 'random_search_logistic_regression.pkl')}
                                                   },
                                              },
                      'svc': {'n_runs': params.svc_random_search_runs,
                              'callbacks': {'neptune_monitor': {'name': 'svc'},
                                            'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                         'random_search_svc.pkl')}
                                            },
                              },
                      },

    'clipper': {'min_val': 0,
                'max_val': 1
                },

    'groupby_aggregation': {'groupby_aggregations': AGGREGATION_RECIPIES
                            },
})
